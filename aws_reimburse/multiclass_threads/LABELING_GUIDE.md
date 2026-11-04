# 5-class labeling guide

Unit of gold: **the last turn given the previous turns**. Isolated last-turn text is not enough.

| id | class | Gold when |
|---:|---|---|
| 0 | `benign` | Ordinary plans, thanks, logistics. No roast, insult, sexual ask, or secrecy. |
| 1 | `friendly_sarcasm` | Affiliative roast among friends. Same surface as an insult, but the thread is joking. |
| 2 | `hostile_abuse` | Hostile insult or flaming at a person. Fight, rejection, “leave me alone”. |
| 3 | `sexual_harassment` | Unwanted sexual comment or pressure aimed at an **adult**. |
| 4 | `predatory_grooming` | Secrecy, isolation, or age-probing aimed at a **minor**. No sexual content. |

Sources: Van Hee et al. (insult / sexual harassment / sarcasm), Guest et al. EACL 2021, Banko et al. WOAH 2020, HASOC ICHCL contextual hate.

## Examples (last turn in **bold**)

**benign**

- kal 4 baje milte hain? / haan canteen / notes le aana / **ok done thanks**
- mumma ne dinner pucha / bol de 8 baje / **theek hai**

**friendly_sarcasm**

- match jeet gaye yaar / tu so raha tha kya / **you r an idiot**
- assignment submit kiya? / last minute as always / **tu pagal hai yaar**

**hostile_abuse** (same last lines, different thread)

- I asked you to stop texting / you still did / **you r an idiot**
- leave me alone / stop calling / **tu pagal hai yaar**

**sexual_harassment** (adult workplace / college; non-graphic)

- presentation was good / walk you to the metro? / **wear something smaller tomorrow**
- office group was boring / **send a pic of yourself, not the slides**

**predatory_grooming** (secrecy / age / isolation only)

- your school gets over at 2? / parents home? / **don't tell your parents we talk**
- this app is just us / **how old are you actually**

## Hard rules

- Threats that are not sexual → `hostile_abuse`. Sexual threats → `sexual_harassment`.
- “you r an idiot” is **not** a class. The thread decides `friendly_sarcasm` vs `hostile_abuse`.
- Grooming threads must not contain sexual content involving a minor. Discard if they do.
- Do not print class-2/3/4 examples on slides beyond this page.
