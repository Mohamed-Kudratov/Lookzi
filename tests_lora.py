"""Test the LoRA prefix/alpha handling without torch or a GPU."""
import ast, pathlib, sys

src = pathlib.Path(r"D:\projects\lvton\pipeline.py").read_text(encoding="utf-8")
tree = ast.parse(src)

wanted = {"_strip_lora_prefix", "_lora_config_from_state_dict"}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
prefixes = next(n for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "_LORA_PREFIXES")

class Cfg:
    def __init__(self, **kw):
        self.r = kw["r"]; self.lora_alpha = kw["lora_alpha"]; self.target_modules = kw["target_modules"]
class T:
    def __init__(self, *shape, val=None): self.shape = shape; self._v = val
    def numel(self): return 1 if self._v is not None else 999
    def item(self): return self._v

ns = {"LoraConfig": Cfg}
exec(compile(ast.Module(body=[prefixes] + nodes, type_ignores=[]), "x", "exec"), ns)
strip, cfg_from = ns["_strip_lora_prefix"], ns["_lora_config_from_state_dict"]

fails = 0
def check(name, cond, detail=""):
    global fails
    print(f"{'ok  ' if cond else 'FAIL'} {name}{'  ' + detail if detail and not cond else ''}")
    fails += not cond

# --- prefix handling ------------------------------------------------------
diffusers_fmt = {"transformer.blocks.0.attn.to_q.lora_A.weight": T(64, 3072)}
sd, p = strip(diffusers_fmt)
check("diffusers prefix stripped", p == "transformer." and "blocks.0.attn.to_q.lora_A.weight" in sd)

comfy_fmt = {"diffusion_model.blocks.0.attn.to_q.lora_A.weight": T(64, 3072)}
sd, p = strip(comfy_fmt)
check("comfyui prefix stripped", p == "diffusion_model.")

bare_fmt = {"blocks.0.attn.to_q.lora_A.weight": T(64, 3072)}
sd, p = strip(bare_fmt)
check("bare keys accepted", p == "" and len(sd) == 1)

try:
    strip({"some.random.weight": T(4)})
    check("non-LoRA rejected", False, "should have raised")
except ValueError as e:
    check("non-LoRA rejected", "no recognisable LoRA" in str(e))

# --- rank / alpha / targets ----------------------------------------------
sd = {
    "blocks.0.attn.to_q.lora_A.weight": T(64, 3072),
    "blocks.0.attn.to_q.lora_B.weight": T(3072, 64),
    "blocks.0.attn.to_out.0.lora_A.weight": T(64, 3072),
    "blocks.0.ff.net.2.lora_A.weight": T(64, 3072),
}
c = cfg_from(sd)
check("rank read from tensor", c.r == 64)
check("alpha defaults to rank", c.lora_alpha == 64)
check("indexed module keeps index", "to_out.0" in c.target_modules and "net.2" in c.target_modules)

# alpha present in the file must win over the rank default
sd_alpha = dict(sd, **{"blocks.0.attn.to_q.alpha": T(val=32.0)})
c2 = cfg_from(sd_alpha)
check("explicit alpha honoured", c2.lora_alpha == 32.0, f"got {c2.lora_alpha}")

# scale applies on top of the file's alpha, not the rank
c3 = cfg_from(sd_alpha, alpha_scale=0.5)
check("alpha_scale multiplies alpha", c3.lora_alpha == 16.0, f"got {c3.lora_alpha}")

print(f"\n{'all passed' if not fails else str(fails) + ' FAILED'}")
sys.exit(1 if fails else 0)
