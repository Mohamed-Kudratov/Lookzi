# Competitive landscape

Researched 2026-08-14. Figures for private companies are self-reported or from
funding announcements; treat revenue claims as directional.

## The one number that matters

**FASHN.ai sells try-on at $0.075/image list, dropping below $0.04 at volume.**

Our measured compute cost on the current model is **$0.066/image**. At the market
price that is roughly a cent of margin at list, and **negative margin** at volume
pricing — before any of the cost of running a company.

With the Lightning 8-step LoRA that cost falls roughly 10× to ~$0.007/image.
That is the whole difference between a business and an experiment. The
optimisation work is not a nice-to-have; it is the viability condition.

## Service 1 — live try-on: the hard side

**Google gave it away.** On 2026-04-30 Google shut down its standalone Doppl app
and moved virtual try-on into Search, Shopping and AI Mode. It is free, it works
from a **single selfie** (since Dec 2025), and it covers billions of listings
from Macy's, Kohl's, Walmart and Nordstrom.

Every giant has already in-housed the capability:

| Buyer | What they bought | When |
|---|---|---|
| Walmart | Zeekit (~$200M) | 2021 |
| Snap | Fit Analytics, Vertebrae | 2021 |
| Meta | Presize | 2022 |
| Zalando | Fashwell | — |
| Browzwear | Lalaland.ai | 2025 |

Independent players:

| Company | Where | Raised | Status |
|---|---|---|---|
| **FASHN.ai** | Tel Aviv, 2022 | ~$2M / largely self-funded | alive, sets the price floor; open-sourced v1.5 under Apache 2.0 |
| **Doji** | US, 2024 | **$14M seed** (Thrive Capital, Seven Seven Six) | well funded, consumer app with AI avatars |
| **Veesual** | Paris, 2020 | $7.5M seed (AXA VP, Techstars) | enterprise, ~20 staff, US/EU/AU clients |

Read that table honestly: FASHN open-sourced its model and prices at $0.04, and
Google does it for nothing. **A generic try-on API sold to Western online retail
is not a business we can win.**

## Service 2 — AI photo studio: the healthier side

Not commoditised by Google, and there is direct evidence of demand:

| Company | Raised | Signal |
|---|---|---|
| **Botika** | **$18M** total ($8M seed; Stardom, Secret Chord, Seedcamp) | 3,000+ brands; **9× revenue and 11× customers** year over year |
| Lalaland.ai | $4M over 6 years | **acquired by Browzwear, 2025** — absorbed rather than scaled |
| Veesual | $7.5M | also sells "Switch Model" imagery |

Botika's growth is the strongest signal in this whole survey: brands will pay
for on-model imagery, and they will pay repeatedly. Lalaland is the cautionary
half — six years, $4M, and an acqui-hire outcome. Being good at the model is not
the same as being good at the business.

The underlying driver is returns. Retail calls them the "silent killer" of
margin; Catches claims a 10% conversion lift and 20–30× ROI for brand partners.
That is the number a seller actually buys.

## Where we are not commoditised

Google's try-on runs on **Google's** surfaces, against **Google's** catalogue.
It does not appear inside a merchant's own product page, and it does not serve a
seller running a shop through Telegram or Instagram in Uzbekistan or the wider
CIS. Neither do Botika, Veesual or Doji — none of them are localised here.

That geography is a real moat while it lasts, and it is the one advantage that
does not depend on having a better model than Google.

## What this implies

1. **Do not sell a generic try-on API to Western retail.** The price is set at
   $0.04 by a competitor who open-sourced their model, and the incumbent gives
   it away.
2. **Lead with Service 2.** Botika proves demand, Google does not compete there,
   and the margin per job is thick enough to absorb a heavy model.
3. **Ship Service 1 as part of a local product**, not as an API — bundled with
   sizing, style recommendation and the seller's own storefront, in the market
   where none of these companies operate.
4. **Get the cost per image under a cent** before selling anything at volume.
   That is the Lightning work, and it is a precondition, not an optimisation.
5. **Sell the returns number, not the technology.** Nobody buys a diffusion
   model; they buy fewer returns and higher conversion.

## Sources

- [Google Shopping AI Mode and virtual try-on](https://blog.google/products-and-platforms/products/shopping/google-shopping-ai-mode-virtual-try-on-update/)
- [Google's try-on expands to more countries](https://techcrunch.com/2025/10/08/googles-virtual-try-on-shopping-tool-expands-to-more-countries-now-lets-you-try-on-shoes/)
- [CNBC — 'Silent killers': AI and retail's returns problem](https://www.cnbc.com/2026/04/05/ai-retail-start-ups-virtual-try-on-tech-margins.html)
- [Doji raises $14M](https://techcrunch.com/2025/05/15/doji-raises-14m-to-make-virtual-try-ons-fun-through-ai-avatars)
- [Botika raises $8M](https://app.dealroom.co/news/feed/botika-raises-8m-for-ai-fashion-models)
- [Browzwear acquires Lalaland](https://thenextweb.com/news/browzwear-snaps-up-dutch-ai-fashion-model-startup-lalaland)
- [Veesual raises $7.5M](https://www.prnewswire.com/news-releases/ai-powered-virtual-try-on-technology-platform-for-the-fashion-industry-veesual-raises-7-5-million-announces-us-expansion-with-new-eileen-fisher-partnership-302119247.html)
- [FASHN API pricing](https://fashn.ai/products/api)
- [Walmart / Zeekit](https://www.retaildive.com/news/will-virtual-fitting-rooms-push-walmart-to-the-fashion-forefront/602245/)
