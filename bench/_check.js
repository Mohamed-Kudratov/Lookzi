
(function () {
"use strict";
var $ = function (id) { return document.getElementById(id); };

function esc(v) {
  return String(v == null ? "" : v)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

var CLIENT = localStorage.getItem("lookzi-client");
if (!CLIENT) {
  CLIENT = "web-" + (crypto.randomUUID ? crypto.randomUUID()
    : Date.now() + Math.random().toString(16).slice(2));
  localStorage.setItem("lookzi-client", CLIENT);
}
try {
  var savedTheme = localStorage.getItem("lookzi-theme");
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;
} catch (e) {}

function api(path, opts) {
  opts = opts || {};
  return fetch(path, {
    method: opts.method || "GET",
    headers: {"Content-Type": "application/json", "X-Client-Id": CLIENT},
    body: opts.body ? JSON.stringify(opts.body) : undefined
  }).then(function (r) {
    return r.json().catch(function () { return {}; }).then(function (d) {
      if (!r.ok) throw new Error(d.detail || ("request failed (" + r.status + ")"));
      return d;
    });
  });
}

var toastTimer;
function toast(msg, bad) {
  var el = $("toast");
  el.textContent = msg;
  el.classList.toggle("bad", !!bad);
  el.classList.add("on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(function () { el.classList.remove("on"); }, 3800);
}

// ---------------------------------------------------------------- state
var TOOLS = [], MODELS = [], job = null, poll = null, result = null, recent = [];
var pick = {tool: null, prompt: "",
            garment_key: null, garment_url: null, garment_kind: null,
            person_key: null, person_url: null,
            model_id: null,
            gender: "woman", age: "20s", build: "average", look: "uzbek",
            modest: "false"};

function tool() { return TOOLS.filter(function (t) { return t.id === pick.tool; })[0]; }
function needs() { return (tool() || {needs: []}).needs; }
function wants(n) {
  var ns = needs();
  return ns.indexOf(n) >= 0 || ns.indexOf(n + "?") >= 0;
}

var ICON = {
  "product-to-model": '<rect x="3" y="4" width="7" height="7" rx="1"/><path d="M17 4v4M15 6h4"/><path d="M7 15v5M14 20v-5a3 3 0 0 1 6 0v5"/>',
  "virtual-try-on": '<path d="M8 3h8l4 4-3 2v12H7V9L4 7z"/>',
  "model-swap": '<circle cx="8" cy="8" r="3"/><circle cx="16" cy="16" r="3"/><path d="M14 6h5v5M10 18H5v-5"/>',
  "product-in-scene": '<path d="M4 5h16v14H4z"/><path d="m4 15 4-4 4 3 3-2 5 4"/><circle cx="9" cy="9" r="1.4"/>',
  "packshot": '<path d="M21 8 12 3 3 8l9 5 9-5z"/><path d="M3 8v8l9 5 9-5V8"/>',
  "model-creation": '<circle cx="12" cy="8" r="4"/><path d="M5 21c0-4 3-6 7-6s7 2 7 6"/><path d="M19 3v4M17 5h4"/>',
  "short-video": '<rect x="3" y="6" width="13" height="12" rx="2"/><path d="m16 10 5-3v10l-5-3z"/>'
};
function svg(id) {
  return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    (ICON[id] || ICON.packshot) + '</svg>';
}

var SLOT = {
  garment: {label: "Garment", hint: "the item, laid flat"},
  person: {label: "Your photo", hint: "full body, facing the camera"},
  model: {label: "Model", hint: "tap to choose who wears it"}
};
var PROMPT_HINT = {
  "product-in-scene": "Describe the shot — the person, the place, the light…",
  "model-creation": "Optional: describe the model in your own words",
  "": "Optional: anything to say about the shot"
};
var EXAMPLES = [
  "a man in his 30s walking among trees in autumn, golden leaves falling around him",
  "a young woman on a Tashkent street in the evening, warm shop lights behind her",
  "in a bright airy room, morning light through a window"
];

// ---------------------------------------------------------------- the bar
function drawPills() {
  $("pills").innerHTML = TOOLS.map(function (t) {
    return '<button class="pill" role="tab" data-id="' + esc(t.id) + '" ' +
      'aria-pressed="' + (t.id === pick.tool) + '"' + (t.ready ? "" : " disabled") + '>' +
      svg(t.id) + esc(t.label) + (t.ready ? "" : " <small>soon</small>") + '</button>';
  }).join("");
}

function drawBar() {
  var t = tool();
  if (!t) return;
  $("tool-name").textContent = t.label;
  $("prompt").placeholder = PROMPT_HINT[t.id] || PROMPT_HINT[""];

  var extras = [];
  if (wants("model")) {
    var m = MODELS.filter(function (x) { return x.id === pick.model_id; })[0];
    extras.push('<button class="mini" data-open="models" aria-pressed="' + !!m + '">' +
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">' +
      '<circle cx="12" cy="8" r="4"/><path d="M5 21c0-4 3-6 7-6s7 2 7 6"/></svg>' +
      (m ? esc(m.display_name) : "Choose a model") + '</button>');
  }
  if (wants("look")) {
    extras.push('<button class="mini" data-open="look">' +
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">' +
      '<path d="M4 6h16M4 12h16M4 18h10"/></svg>' +
      esc(pick.gender + ", " + pick.age + ", " + pick.look) + '</button>');
  }
  if (ASKS_KIND[t.id] && pick.garment_key && pick.garment_kind) {
    extras.push('<button class="mini" data-open="kind" aria-pressed="true">' +
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">' +
      '<path d="M20 6 9 17l-5-5"/></svg>' + esc(kindLabel(pick.garment_kind)) +
      '</button>');
  }
  if (t.id === "product-in-scene" || t.id === "model-creation") {
    extras.push('<button class="mini" data-example="1">' +
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">' +
      '<path d="M12 3v18M3 12h18"/></svg>Example</button>');
  }
  $("extras").innerHTML = extras.join("");

  var missing = [];
  needs().forEach(function (n) {
    if (n.slice(-1) === "?") return;
    if (n === "garment" && !pick.garment_key) missing.push("a garment photo");
    if (n === "person" && !pick.person_key) missing.push("your photo");
    if (n === "model" && !pick.model_id) missing.push("a model");
    if (n === "prompt" && !pick.prompt.trim()) missing.push("a description");
  });
  $("run").disabled = missing.length > 0 || !!job;
  $("run-label").textContent = "Run · " + t.cost + " credit" + (t.cost === 1 ? "" : "s");
  $("hint").textContent = missing.length
    ? "Needs " + missing.join(" and ")
    : (t.typical_seconds ? "usually about " + Math.round(t.typical_seconds) + "s" : "");
}

// ---------------------------------------------------------------- the stage
function imageSlots() {
  return needs().filter(function (n) { return SLOT[n]; });
}

function drawStage() {
  if (job) return;                    // the veil owns the stage while it runs
  if (result) return drawResult();
  var slots = imageSlots();
  if (!slots.length) return drawBlank();
  $("stage").innerHTML = '<div class="cards">' + slots.map(card).join("") + '</div>';
}

function card(need) {
  var spec = SLOT[need];
  var killer = '<button class="kill" data-clear="' + esc(need) + '" aria-label="Remove">' +
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4">' +
    '<path d="M18 6 6 18M6 6l12 12"/></svg></button>';
  if (need === "model") {
    var m = MODELS.filter(function (x) { return x.id === pick.model_id; })[0];
    return '<div class="card' + (m ? "" : " want") + '" data-open="models">' +
      '<span class="label">' + esc(spec.label) + '</span>' +
      (m && m.preview
        ? '<img src="' + esc(m.preview) + '" alt="' + esc(m.display_name) + '">' + killer
        : '<span class="drop"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="12" cy="8" r="4"/><path d="M5 21c0-4 3-6 7-6s7 2 7 6"/></svg>' +
          '<b>Choose a model</b><span>' + esc(spec.hint) + '</span></span>') +
      '</div>';
  }
  var url = pick[need + "_url"];
  return '<label class="card' + (url ? "" : " want") + '" data-need="' + esc(need) + '">' +
    '<span class="label">' + esc(spec.label) + '</span>' +
    '<input type="file" accept="image/*" data-need="' + esc(need) + '">' +
    (url ? '<img src="' + esc(url) + '" alt="">' + killer
         : '<span class="drop"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M12 16V4m0 0L8 8m4-4 4 4"/><path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></svg>' +
           '<b>Drop a photo</b><span>' + esc(spec.hint) + '</span></span>') +
    '</label>';
}

function drawBlank() {
  var t = tool() || {};
  $("stage").innerHTML =
    '<div class="blank">' +
      '<div class="art">' +
        '<span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M8 3h8l4 4-3 2v12H7V9L4 7z"/></svg></span>' +
        '<span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="8" r="4"/><path d="M5 21c0-4 3-6 7-6s7 2 7 6"/></svg></span>' +
        '<span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 5h16v14H4z"/><path d="m4 15 4-4 4 3 3-2 5 4"/></svg></span>' +
      '</div>' +
      '<h2>' + esc(t.label || "Studio") + '</h2>' +
      '<p>' + esc(t.blurb || "") + '</p>' +
    '</div>';
}

// Which of the job's pictures is on screen, and the buttons to change it.
// One result draws no buttons: a control with a single option only takes room.
function shown() {
  var v = result.variants || [];
  return (v.length > 1 && v[result.at]) ? v[result.at].url : result.url;
}
var VARIANT_NAME = {packshot: "Retouched", cutout: "Cut out only"};
function variantPicker() {
  var v = result.variants || [];
  if (v.length < 2) return "";
  var flag = "", note = v[result.at] && v[result.at].notes;
  if (note && note.failed && note.failed.length) {
    // The gate's opinion, said plainly and left as the seller's decision.
    // They have the garment in their hand and we do not.
    flag = '<span class="flag">check the ' + esc(note.failed.join(" and ")) +
      ' — it may not match the real garment</span>';
  }
  return '<div class="pick">' + v.map(function (x, i) {
    return '<button data-variant="' + i + '" aria-pressed="' + (i === result.at) +
      '">' + esc(VARIANT_NAME[x.variant] || x.variant || "Result") + '</button>';
  }).join("") + flag + '</div>';
}

// What a finished picture is good for. A packshot is a garment, so it feeds
// anything that wants a garment; a try-on or a product shot is a dressed
// person, which is what "change the model" reads the clothes from; a model
// made to order is a person to dress.
//
// The slot matters as much as the tool. "Change the model" takes the customer's
// photograph in the garment slot -- the clothes on the person are the point of
// it -- and sending it as a person would silently do nothing.
var ONWARD = {
  "packshot": [["virtual-try-on", "garment"], ["product-to-model", "garment"],
               ["product-in-scene", "garment"]],
  "product-to-model": [["model-swap", "garment"]],
  "virtual-try-on": [["model-swap", "garment"]],
  "product-in-scene": [["model-swap", "garment"]],
  "model-swap": [["virtual-try-on", "person"]],
  "model-creation": [["virtual-try-on", "person"]]
};

function onwardRow() {
  var next = ONWARD[result.tool] || [];
  next = next.filter(function (n) {
    return TOOLS.filter(function (t) { return t.id === n[0] && t.ready; }).length;
  });
  if (!next.length || !shownKey()) return "";
  return '<div class="onward"><span>Use this in</span>' + next.map(function (n) {
    var t = TOOLS.filter(function (x) { return x.id === n[0]; })[0];
    return '<button data-onward="' + esc(n[0]) + '" data-slot="' + esc(n[1]) + '">' +
      svg(n[0]) + esc(t.label) + '</button>';
  }).join("") + '</div>';
}

function shownKey() {
  var v = result.variants || [];
  return (v.length && v[result.at || 0] && v[result.at || 0].key) ||
         (v[0] && v[0].key) || null;
}

function drawResult() {
  var ins = [];
  if (result.inputs.garment) ins.push(["Garment", result.inputs.garment]);
  if (result.inputs.person) ins.push(["Your photo", result.inputs.person]);
  if (result.inputs.model) ins.push(["Model", result.inputs.model]);
  $("stage").innerHTML =
    '<div class="done">' +
      '<div class="ins">' +
        '<button class="btn" id="back" style="justify-content:center">' +
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><path d="M19 12H5m0 0 6-6m-6 6 6 6"/></svg>Back</button>' +
        ins.map(function (p) {
          return '<figure><figcaption>' + esc(p[0]) + '</figcaption>' +
            '<img src="' + esc(p[1]) + '" alt=""></figure>';
        }).join("") +
      '</div>' +
      '<div>' + variantPicker() +
      '<div class="big"><img src="' + esc(shown()) + '" alt="The finished image">' +
        '<button class="btn save" id="save">' +
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><path d="M12 4v12m0 0 4-4m-4 4-4-4"/><path d="M5 20h14"/></svg>Save</button>' +
        // A model made to order was a picture in the history and nothing more.
        // The tool promises one that belongs to you; this is what makes it true.
        (result.tool === "model-creation"
          ? '<button class="btn keep" id="keep"' + (result.kept ? " disabled" : "") + '>' +
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9">' +
            '<circle cx="12" cy="8" r="4"/><path d="M5 21c0-4 3-6 7-6s7 2 7 6"/></svg>' +
            (result.kept ? "In your models" : "Keep as my model") + '</button>'
          : "") +
      '</div></div>' +
      onwardRow() +
    '</div>';
}

function drawVeil(secs, state, typical) {
  var over = typical && secs > typical * 1.4;
  $("stage").innerHTML =
    '<div class="cards"><div class="card" style="width:min(420px,84vw);cursor:default">' +
      '<div class="veil"><div>' +
        '<div class="n">' + secs.toFixed(1) + 's</div>' +
        '<div class="s">' + esc(state) + '</div>' +
        '<div class="h">' + (typical
          ? (over ? "longer than usual — normally about " + Math.round(typical) + "s"
                  : "usually about " + Math.round(typical) + "s")
          : "") + '</div>' +
        '<div class="track"><i style="width:' +
          (typical ? Math.min(96, (secs / typical) * 100) : 0) + '%"></i></div>' +
        '<button class="btn" id="cancel-job" style="margin-top:6px">Cancel</button>' +
      '</div></div>' +
    '</div></div>';
}

// ---------------------------------------------------------------- big view
//
// One picture, filling the screen, with a close button -- and it opens here
// rather than in a second tab. A tab is for a document you are going to keep;
// this is a look, and a look should close where it opened.
var view = {frames: [], at: 0};

function paintView() {
  var f = view.frames[view.at];
  if (!f) return;
  $("view-img").src = f.url;
  $("view-cap").innerHTML = '<b>' + esc(f.label) + '</b> · ' +
    (view.at + 1) + " of " + view.frames.length;
  $("view-prev").disabled = view.frames.length < 2;
  $("view-next").disabled = view.frames.length < 2;
}
function stepView(d) {
  if (view.frames.length < 2) return;
  view.at = (view.at + d + view.frames.length) % view.frames.length;
  paintView();
}
function openView(frames, at) {
  view.frames = frames.filter(function (f) { return f && f.url; });
  if (!view.frames.length) return;
  view.at = Math.min(Math.max(at || 0, 0), view.frames.length - 1);
  paintView();
  $("view").showModal();
}

$("view-close").addEventListener("click", function () { $("view").close(); });
$("view-prev").addEventListener("click", function () { stepView(-1); });
$("view-next").addEventListener("click", function () { stepView(1); });
$("view").addEventListener("click", function (e) {
  if (!e.target.closest("img,button")) this.close();
});
document.addEventListener("keydown", function (e) {
  if (!$("view").open) return;
  if (e.key === "ArrowLeft") { e.preventDefault(); stepView(-1); }
  if (e.key === "ArrowRight") { e.preventDefault(); stepView(1); }
});

// Fetched and handed over as a file. A link with `download` on a signed URL
// from another origin is ignored and opens a tab instead, which is the thing
// we are trying not to do.
function save(url) {
  fetch(url).then(function (r) { return r.blob(); }).then(function (b) {
    var a = document.createElement("a");
    a.href = URL.createObjectURL(b);
    a.download = "lookzi-" + Date.now() + ".png";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 4000);
  }).catch(function () { toast("Could not save that one.", true); });
}

// ---------------------------------------------------------------- popovers
var LOOK = [
  ["gender", "Who", [["woman", "Woman"], ["man", "Man"]]],
  ["age", "Age", [["20s", "20s"], ["30s", "30s"], ["40s", "40s"], ["50s", "50s"]]],
  ["build", "Build", [["slim", "Slim"], ["average", "Average"], ["fuller", "Fuller"]]],
  ["look", "From", [["uzbek", "Uzbek"], ["kazakh", "Kazakh"], ["tajik", "Tajik"], ["slavic", "Slavic"]]],
  ["modest", "Dress", [["false", "Everyday"], ["true", "Modest"]]]
];

function openPop(which) {
  var html;
  if (which === "kind") { askKind(null); return; }
  if (which === "models") {
    html = '<h3>Who wears it</h3><div class="grid-models">' + MODELS.map(function (m) {
      return '<button data-model="' + esc(m.id) + '" aria-pressed="' +
        (m.id === pick.model_id) + '">' +
        '<span class="ph">' + (m.preview ? '<img src="' + esc(m.preview) + '" alt="">' : '') + '</span>' +
        '<span class="meta"><b>' + esc(m.display_name) + '</b>' +
        '<span class="age">' + (m.mine ? "yours" : esc(m.age)) + '</span></span></button>';
    }).join("") + '</div>';
  } else {
    html = '<h3>The look</h3><div class="look">' + LOOK.map(function (r) {
      return '<div class="look-row"><span>' + esc(r[1]) + '</span><div class="seg">' +
        r[2].map(function (o) {
          return '<button data-look="' + esc(r[0]) + '" data-v="' + esc(o[0]) + '" ' +
            'aria-pressed="' + (String(pick[r[0]]) === o[0]) + '">' + esc(o[1]) + '</button>';
        }).join("") + '</div></div>';
    }).join("") + '</div>';
  }
  $("pop").innerHTML = html +
    '<button class="icon-btn close" data-close="1" aria-label="Close">' +
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M18 6 6 18M6 6l12 12"/></svg></button>';
  $("pop").hidden = false;
  $("scrim").hidden = false;
}
function closePop() { $("pop").hidden = true; $("scrim").hidden = true; }

// ---------------------------------------------------------------- uploading
function upload(need, file) {
  var type = file.type || "image/jpeg";
  // Shown from the local file immediately: waiting for object storage before
  // anything appears makes a good connection feel slow and a poor one broken.
  pick[need + "_url"] = URL.createObjectURL(file);
  if (need === "garment") pick.garment_kind = null;
  result = null;
  drawStage(); drawBar();
  api("/uploads", {method: "POST", body: {kind: need, content_type: type}})
    .then(function (u) {
      return fetch(u.url, {method: "PUT", body: file, headers: {"Content-Type": type}})
        .then(function (r) {
          if (!r.ok) throw new Error("upload failed (" + r.status + ")");
          pick[need + "_key"] = u.key;
          drawStage(); drawBar();
        });
    })
    .catch(function (e) {
      toast(e.message, true);
      pick[need + "_key"] = null; pick[need + "_url"] = null;
      drawStage(); drawBar();
    });
}

// ------------------------------------------------------------ what is it
// Asked on the button press, not on the upload.
//
// Two reasons it belongs here. The seller is about to act, so they are paying
// attention and cannot forget it the way a field filled in ten minutes ago is
// forgotten. And by this point the tool is known, so the same one interruption
// can carry the warning that belongs to that tool -- asked at upload time we
// would have the category and not yet the tool, and the warning would need a
// second interruption of its own.
//
// Asked once. The answer is remembered against the photograph, server side, so
// a packshot carried into a try-on arrives already answered.
var KINDS = [["upper", "Top", "shirt, jacket, blouse, sweater"],
             ["lower", "Bottom", "trousers, jeans, skirt, shorts"],
             ["overall", "Full-piece", "dress, jumpsuit, gown"]];

// Where the answer is worth having. Not "change the model": what is uploaded
// there is a person already dressed, and asking whether a photograph of
// somebody in an outfit is a top has no answer.
var ASKS_KIND = {"packshot": 1, "product-to-model": 1, "virtual-try-on": 1,
                 "product-in-scene": 1};

// These three put the garment on a body, and the model that does it wears
// everything on the torso -- a skirt comes back worn as a top. It reads the
// garment image and ignores text, so there is no wording that redirects it;
// see docs/CONTROLS.md. Said plainly and up front, and it does not block:
// the seller decides whether to spend the credit.
var TORSO_ONLY = {"product-to-model": 1, "virtual-try-on": 1,
                  "product-in-scene": 1};

function kindLabel(k) {
  var r = KINDS.filter(function (x) { return x[0] === k; })[0];
  return r ? r[1] : null;
}

function askKind(then) {
  $("pop").innerHTML =
    '<h3>What is this garment?</h3>' +
    '<p class="pop-sub">So the packshot names it correctly, and so we can tell ' +
    'you up front where it works. Asked once per photo.</p>' +
    '<div class="kinds">' + KINDS.map(function (k) {
      return '<button data-kind="' + esc(k[0]) + '"' +
        (pick.garment_kind === k[0] ? ' aria-pressed="true"' : '') + '>' +
        '<b>' + esc(k[1]) + '</b><span>' + esc(k[2]) + '</span></button>';
    }).join("") + '</div>' +
    '<button class="icon-btn close" data-close="1" aria-label="Close">' +
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">' +
    '<path d="M18 6 6 18M6 6l12 12"/></svg></button>';
  $("pop").hidden = false;
  $("scrim").hidden = false;
  afterKind = then;
}

// The warning, in the same interruption rather than a second one.
function warnBottom() {
  $("pop").innerHTML =
    '<h3>A bottom will come out worn on top</h3>' +
    '<p class="pop-sub">This tool dresses a body, and the model it uses puts ' +
    'every garment on the torso — a skirt comes back looking like a top. We ' +
    'cannot steer it, so this is what you would get.<br><br>' +
    '<b>Packshot</b> works on bottoms and gives you a clean catalogue photo.</p>' +
    '<div class="warn-row">' +
      '<button class="btn" data-kindgo="1">Run it anyway</button>' +
      '<button class="btn primary" data-kindgo="packshot">Use Packshot</button>' +
    '</div>' +
    '<button class="icon-btn close" data-close="1" aria-label="Close">' +
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">' +
    '<path d="M18 6 6 18M6 6l12 12"/></svg></button>';
  $("pop").hidden = false;
  $("scrim").hidden = false;
}

var afterKind = null;

// ---------------------------------------------------------------- running
function run() {
  var t = tool();
  if (!t) return;
  // The one question, and only when it has not been answered for this photo.
  if (ASKS_KIND[t.id] && pick.garment_key && !pick.garment_kind) {
    askKind(function () {
      if (TORSO_ONLY[t.id] && pick.garment_kind === "lower") return warnBottom();
      run();
    });
    return;
  }
  job = "pending";
  drawBar();
  api("/jobs", {method: "POST", body: {
    tool: t.id,
    // Whatever was uploaded is the garment. For "change the model" that upload
    // is a person wearing clothes, and the clothes are the point of it.
    garment_key: pick.garment_key || (wants("person") ? pick.person_key : null),
    // When one of our models was chosen, the model is who wears it.
    person_key: wants("model") ? null : (pick.person_key || null),
    model_id: pick.model_id || null,
    // What the seller said it is. The try-on model ignores it -- it always
    // has -- but the packshot reads it, and the server remembers it against
    // the photograph so the next tool in the chain does not ask again.
    mode: pick.garment_kind || null,
    prompt: pick.prompt.trim() || null,
    gender: pick.gender, age: pick.age, build: pick.build, look: pick.look,
    modest: pick.modest === "true",
    // A retried request must not become a second job and a second charge.
    idem_key: "web-" + Date.now() + "-" + Math.random().toString(16).slice(2)
  }}).then(function (j) { watch(j.job_id); })
    .catch(function (e) { job = null; toast(e.message, true); drawBar(); drawStage(); });
}

function watch(id) {
  job = id;
  result = null;
  var t0 = Date.now(), typical = (tool() || {}).typical_seconds;
  var inputs = {
    garment: wants("garment") ? pick.garment_url : null,
    person: pick.person_url,
    model: (function () {
      var m = MODELS.filter(function (x) { return x.id === pick.model_id; })[0];
      return m ? m.preview : null;
    })()
  };
  drawVeil(0, "queued", typical);
  clearInterval(poll);
  poll = setInterval(function () {
    api("/jobs/" + id).then(function (j) {
      var secs = (Date.now() - t0) / 1000;
      if (j.status === "done") {
        clearInterval(poll); job = null;
        result = {url: j.result_url, seconds: j.seconds, inputs: inputs,
                  tool: pick.tool, job_id: id, kept: false,
                  variants: (j.results || []).filter(function (r) { return r.url; }),
                  at: 0};
        toast("Done in " + (j.seconds != null ? j.seconds : secs.toFixed(1)) + "s");
        refresh(); drawStage(); drawBar();
        return;
      }
      if (j.status === "failed" || j.status === "cancelled") {
        clearInterval(poll); job = null;
        toast(j.status === "failed" ? "That one failed — the credit came back."
                                    : "Cancelled.", j.status === "failed");
        refresh(); drawStage(); drawBar();
        return;
      }
      drawVeil(secs, j.status === "queued"
        ? (j.position ? j.position + " in the queue" : "queued") : j.status, typical);
    }).catch(function (e) {
      clearInterval(poll); job = null; toast(e.message, true); drawStage(); drawBar();
    });
  }, 1100);
}

// ---------------------------------------------------------------- events
function selectTool(id) {
  var before = pick.tool;
  pick.tool = id;
  if (before !== id) {
    // An upload the new tool has no use for is state that silently changes what
    // gets made; one it still wants is a chore to upload again.
    if (!wants("garment")) { pick.garment_key = null; pick.garment_url = null; }
    if (!wants("person")) { pick.person_key = null; pick.person_url = null; }
    if (!wants("model")) { pick.model_id = null; }
    result = null;
  }
  drawPills(); drawBar(); drawStage();
}

$("pills").addEventListener("click", function (e) {
  var b = e.target.closest(".pill");
  if (!b || b.disabled) return;
  selectTool(b.dataset.id);
});

document.addEventListener("click", function (e) {
  var kill = e.target.closest("[data-clear]");
  if (kill) {
    e.preventDefault(); e.stopPropagation();
    var n = kill.dataset.clear;
    if (n === "model") pick.model_id = null;
    else {
      pick[n + "_key"] = null; pick[n + "_url"] = null;
      // A different photograph is a different garment. Carrying the last
      // answer over would silently label it, which is the one thing asking
      // was meant to prevent.
      if (n === "garment") pick.garment_kind = null;
    }
    drawStage(); drawBar();
    return;
  }
  if (e.target.closest("[data-close]") || e.target.id === "scrim") {
    // Closing the question is a decision not to run, not a decision to run
    // without answering it.
    afterKind = null;
    closePop(); return;
  }

  var kd = e.target.closest("[data-kind]");
  if (kd) {
    pick.garment_kind = kd.dataset.kind;
    closePop(); drawBar();
    var then = afterKind; afterKind = null;
    if (then) then();
    return;
  }

  var kg = e.target.closest("[data-kindgo]");
  if (kg) {
    closePop();
    if (kg.dataset.kindgo === "packshot") {
      pick.tool = "packshot";
      drawStage(); drawBar();
      toast("Switched to Packshot — press Generate when you are ready.");
      return;
    }
    run();
    return;
  }

  var m = e.target.closest("[data-model]");
  if (m) { pick.model_id = m.dataset.model; closePop(); drawStage(); drawBar(); return; }

  var lk = e.target.closest("[data-look]");
  if (lk) {
    pick[lk.dataset.look] = lk.dataset.v; openPop("look"); drawBar(); return;
  }

  var open = e.target.closest("[data-open]");
  if (open) { openPop(open.dataset.open); return; }

  if (e.target.closest("[data-example]")) {
    pick.prompt = EXAMPLES[Math.floor(Math.random() * EXAMPLES.length)];
    $("prompt").value = pick.prompt; drawBar(); $("prompt").focus();
    return;
  }
  if (e.target.closest("#back")) { result = null; drawStage(); drawBar(); return; }

  if (e.target.closest(".big img") && result) {
    // Everything that made this picture, in the order it happened.
    var f = [];
    if (result.inputs.person) f.push({url: result.inputs.person, label: "your photo"});
    if (result.inputs.garment) f.push({url: result.inputs.garment, label: "product"});
    if (result.inputs.model) f.push({url: result.inputs.model, label: "model"});
    f.push({url: result.url, label: "result"});
    openView(f, f.length - 1);
    return;
  }
  var pv = e.target.closest("[data-variant]");
  if (pv && result) { result.at = Number(pv.dataset.variant); drawResult(); return; }
  var on = e.target.closest("[data-onward]");
  if (on && result) {
    var slot = on.dataset.slot;
    // The result becomes the input. No download, no second upload: the object
    // is already in storage and the next job can name it.
    pick[slot + "_key"] = shownKey();
    pick[slot + "_url"] = shown();
    if (slot === "garment") {
      pick.person_key = null; pick.person_url = null;
      // A packshot of a skirt is still a skirt. The server says so, having
      // been told once, so the seller is not asked the same question at every
      // hop of a three-tool job.
      var v = (result.variants || [])[result.at || 0] || (result.variants || [])[0];
      pick.garment_kind = (v && v.garment_kind) || null;
    }
    pick.tool = on.dataset.onward;
    result = null;
    drawStage(); drawBar();
    toast("Carried over — add whatever else it needs and run.");
    return;
  }
  if (e.target.closest("#save") && result) { save(shown()); return; }

  if (e.target.closest("#keep") && result && result.job_id) {
    api("/models/keep", {method: "POST", body: {job_id: result.job_id}})
      .then(function (m) {
        result.kept = true;
        // Chosen straight away: keeping a model and then having to go and find
        // it in the picker is two steps where the point was one.
        pick.model_id = m.id;
        return api("/models").then(function (rows) { MODELS = rows; });
      })
      .then(function () { toast("Kept — it is in your models now."); drawResult(); drawBar(); })
      .catch(function (err) { toast(err.message, true); });
    return;
  }

  if (e.target.closest("#cancel-job") && job && job !== "pending") {
    api("/jobs/" + job + "/cancel", {method: "POST"})
      .then(function () { toast("Cancelled — nothing charged."); })
      .catch(function (err) { toast(err.message, true); });
  }
});

$("stage").addEventListener("change", function (e) {
  var input = e.target.closest("input[type=file]");
  if (input && input.files[0]) upload(input.dataset.need, input.files[0]);
});
["dragenter", "dragover"].forEach(function (ev) {
  $("stage").addEventListener(ev, function (e) {
    var c = e.target.closest(".card[data-need]"); if (!c) return;
    e.preventDefault(); c.classList.add("over");
  });
});
["dragleave", "drop"].forEach(function (ev) {
  $("stage").addEventListener(ev, function (e) {
    var c = e.target.closest(".card[data-need]"); if (!c) return;
    e.preventDefault(); c.classList.remove("over");
  });
});
$("stage").addEventListener("drop", function (e) {
  var c = e.target.closest(".card[data-need]"); if (!c) return;
  if (e.dataTransfer.files[0]) upload(c.dataset.need, e.dataTransfer.files[0]);
});

$("prompt").addEventListener("input", function () {
  pick.prompt = this.value;
  this.style.height = "auto";
  this.style.height = Math.min(120, this.scrollHeight) + "px";
  drawBar();
});
$("prompt").addEventListener("keydown", function (e) {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && !$("run").disabled) run();
});
$("run").addEventListener("click", run);
$("theme").addEventListener("click", function () {
  var next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("lookzi-theme", next); } catch (e) {}
});
$("nav-models").addEventListener("click", function () { openPop("models"); });

// The gallery is the owner's, and behind a key. It is carried from the URL the
// studio was opened with and remembered, so opening /?key=... once is enough
// and a visitor who has the plain link never sees anybody's uploads.
(function () {
  var k = new URLSearchParams(location.search).get("key");
  try {
    if (k) localStorage.setItem("lookzi-key", k);
    else k = localStorage.getItem("lookzi-key");
  } catch (e) {}
  if (k) $("nav-gallery").href = "/review?key=" + encodeURIComponent(k);
})();

// ---------------------------------------------------------------- loading
function refresh() {
  api("/me").then(function (m) {
    // An unlimited account has no ledger to read from, and "0 credits" on the
    // account that is meant to be able to run anything reads as broken.
    $("credits").textContent = m.plan === "unlimited" ? "∞" : m.credits;
  })
    .catch(function () {});
  api("/jobs?limit=12").then(function (rows) {
    recent = rows.filter(function (r) { return r.result_url; });
    if (result) drawResult();
  }).catch(function () {});
}

function pulse() {
  api("/health").then(function (h) {
    $("gpu").dataset.s = h.workers ? "ok" : "none";
    $("gpu-text").textContent = h.workers
      ? h.workers + " worker" + (h.workers === 1 ? "" : "s") + " ready" +
        (h.queued ? " · " + h.queued + " waiting" : "")
      : "no worker running";
  }).catch(function () {
    $("gpu").dataset.s = "none";
    $("gpu-text").textContent = "cannot reach the service";
  });
}

Promise.all([api("/tools"), api("/models")])
  .then(function (r) {
    TOOLS = r[0]; MODELS = r[1];
    var first = TOOLS.filter(function (t) { return t.ready; })[0] || TOOLS[0];
    pick.tool = first.id;
    drawPills(); drawBar(); drawStage();
    refresh(); pulse();
    setInterval(pulse, 6000);
  })
  .catch(function (e) {
    $("stage").innerHTML = '<div class="blank"><h2>The service is not answering</h2>' +
      '<p>' + esc(e.message) + '</p></div>';
  });
})();
