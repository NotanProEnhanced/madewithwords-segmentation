# Biometric consent records stamped `bio-v1-2026-06` (corrected 2026-09-02)

A dated record of a discrepancy between the consent notice customers were shown and the
version string stored alongside their consent. Written because the affected records cannot
be corrected — rewriting a consent log would be worse than the error — so the discrepancy
needs to exist somewhere findable instead of being rediscovered later by someone else.

## What happened

Each stored consent record carries a `consent_version` (`app/main.py:1773`), whose purpose is
stated in `app/config.py:252`:

> Versioned so each stored consent record points at the exact notice wording the user saw.
> Bump when the upload-time biometric notice copy materially changes.
> v2 adds the age (13+) + parent/guardian attestation folded into the same box.

The value has two defaults. `app/config.py:255` read `bio-v3-2026-06`. `docker-compose.yml`
defaulted it to `bio-v1-2026-06`, and compose wins: it places the variable in the container's
environment, so the code's default was never reached. No `.env` set it, so every container
stamped `bio-v1-2026-06`.

The notice actually displayed at upload (`static/index.html:609`) reads:

> I'm 13+, I have the right to use this photo, and I consent to the facial-geometry analysis
> that creates the artwork — no faceprint is kept and the photo is auto-deleted.

That text contains the 13+ attestation, which the config comment attributes to **v2**. So the
notice on screen was at least v2, the code intended v3, and the records say v1.

**The wording shown to customers was not wrong. The version stamp on the record was.** Each
record understates the notice its signer actually agreed to.

## Scope

Counted 2026-09-02 across all five trees:

    bio-v1-2026-06     2263

That is every biometric consent record in existence at that date.

## What was done

`docker-compose.yml` corrected to `${TYPO_BIO_CONSENT_VERSION:-bio-v3-2026-06}`, matching the
code, on 2026-09-02. Records written from that deploy onward carry the correct string.

Existing records were **not** modified. Their `consent_version` field still reads
`bio-v1-2026-06`, and this note is the correspondence between that string and the notice those
customers were actually shown.

## How it was found, and what prevents a repeat

Found by `ops/env-default-audit.py`, which compares every `${VAR:-default}` in
`docker-compose.yml` against the code's own default and reports disagreements. It was written
after the same class of mismatch — `TYPO_POLARITY` reading `0` in code while compose defaulted
it to `1` — cost an afternoon of debugging a render, because reading the code described a
program that was not the one running.

Run it before any release:

    ./ops/env-default-audit.py            # full report
    ./ops/env-default-audit.py --quiet    # disagreements only; exits 1 if any

One disagreement is deliberately still open at the time of writing: `TYPO_PRICE_CENTS`,
compose `900` against code `1499`, with every `.env` setting `2900`. Live pricing is correct;
a new tree without that `.env` line would sell at $9.00.
