# How customers reach us

The channel question has no single answer because there is no single customer.
Three segments, three channels, one backend.

## The rule that keeps this cheap

**API-first internally, channels as clients.**

The Telegram bot, the web workspace and the public API are all callers of the
same `/v1/jobs` surface. Build the API first — not to sell it, but because every
channel after it becomes a frontend rather than a system. Get this wrong and
each channel becomes its own backend, and the third one is never built.

FASHN arrived at a unified workspace *last*, after the API and the individual
tools. That order is not an accident and it is the order in `ARCHITECTURE.md`.

## The segments

### A — Social sellers. Telegram. **Start here.**

The largest segment by headcount in Uzbekistan, and the one every Western
competitor ignores. These sellers run their shop inside Telegram or Instagram.
They are not technical, they will never call an API, and they will not install
software. They photograph a garment on a phone and post it.

**Channel: a Telegram Mini App.**

The seller sends a product photo into a chat and gets catalogue-grade images
back. No account creation ceremony, no integration, no learning curve — the
product lives where their business already lives. A Mini App is a full web view
inside Telegram, so the same UI we build for the web works here with a different
shell.

This is the channel with the lowest friction and the largest addressable count,
and it is unreachable by Google, Botika or FASHN — not because they cannot build
it, but because they are not in this market.

### B — Small and medium online stores. Web platform.

Own site or a marketplace storefront. Semi-technical, sometimes with a
developer. Budget larger than segment A, count much smaller.

**Channel: the web workspace** — the single prompt box, gallery and elements
library from `ARCHITECTURE.md`. Same backend, different shell from the Mini App.

### C — Brands and marketplaces. API.

Have engineers, want it inside their own product, care about volume pricing and
uptime. Highest revenue per account, smallest count.

**Channel: the documented public API.** It already exists by then — this is
opening the door, not building the house.

## The lever worth more than all three

**Marketplace integration.**

If a local marketplace offers its sellers "generate professional photos for this
listing" and that button is powered by us, distribution to every seller on that
platform arrives at once. One integration replaces years of one-by-one
acquisition.

That is a B2B2C play and it needs a pilot with real numbers — conversion lift
and returns reduction on a few hundred listings — before anyone will sign it. It
should be pursued early even though it lands late, because the sales cycle is
long.

The number to bring to that meeting is not image quality. It is **fewer returns
and higher conversion**, which is what the CNBC reporting calls retail's "silent
killer" — a claimed 10% conversion lift and 20–30× ROI is what makes the case.

## MCP server

Honest assessment: it costs about a day on top of a finished API and is worth
doing — but for positioning, not revenue. No clothing seller in Tashkent will
discover this product through an MCP registry. Build it after the API is public,
treat it as presence in the AI-tool ecosystem, and do not let it compete for
attention with the Telegram work.

## What the market demands that the competitors do not do

**Language.** Uzbek and Russian, in the UI and in the agent router. Every
competitor here is English-first. A seller writing *"bu ko'ylakni modelga
kiydir"* must work exactly as well as the English equivalent. The router is a
small LLM call, so this costs almost nothing and cannot be matched without
entering the market.

**Payment.** Local rails — Click, Payme, Uzcard/Humo — not international cards.
Prepaid credits with small top-ups rather than monthly subscriptions, because
price sensitivity is high and committing to a recurring charge is a much bigger
decision here than topping up a balance.

**Faces.** The roster should look like the customers being sold to. Western
tools generate Western models; a seller in Tashkent converts better with a model
their buyer recognises. This costs nothing but the choice to do it.

**Price.** FASHN's $0.04–0.075 per image is a Western price. The local market
will need lower, which is exactly why the cost work in `STRATEGY.md` is a
precondition — at $0.066 of GPU per image there is no room to price for this
market at all.

## Sequence

1. **Telegram Mini App** — lowest friction, largest count, unreachable by
   incumbents. Ship first.
2. **Web workspace** — same backend, for sellers who outgrow the bot.
3. **Public API** — document what already exists; brands and integrations.
4. **Marketplace pilot** — start conversations early, expect it to close late.
5. **MCP server** — a day's work, for positioning.

Geographic order follows the same logic: Uzbekistan, then Kazakhstan and the
rest of Central Asia, then the wider CIS. Telegram commerce, price sensitivity
and Russian-language support carry across all of them, which means the product
built for the first market is already most of the way to the others.
