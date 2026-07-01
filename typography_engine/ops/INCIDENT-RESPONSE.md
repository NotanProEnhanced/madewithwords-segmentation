# Incident Response & Breach Notification

How we detect, contain, assess, and — when required — report a security incident
or personal-data breach for Typortrait™ / Loved in Words™. Aligned to **PIPEDA**
(Canada), **Quebec Law 25**, and **GDPR** (EU/UK) where applicable.

> Not legal advice. Confirm exact notification wording, thresholds, and deadlines
> with privacy counsel at the time of an incident.

**Incident Lead / Privacy Officer:** the founder — `support@typortrait.com`. *(Fill in name.)*

---

## 1. Definitions & severity
- **Security incident** — any event that may compromise the confidentiality, integrity, or availability of the Service or its data (outage, malware, unauthorized access, lost/leaked credential, a sub-processor breach).
- **Personal-data breach** — an incident involving unauthorized access, disclosure, loss, or destruction of personal information (uploaded photos, emails, order or consent data).
- **Severity:** **SEV-1** confirmed personal-data breach or full outage · **SEV-2** suspected breach / partial outage / credential exposure · **SEV-3** minor, no personal-data impact.

## 2. Response lifecycle — Detect → Contain → Assess → Notify → Recover → Review

### Detect
Sources: uptime + backup-failure alerts; error logs; a report from a user or partner; a notice from a sub-processor (Stripe, Printful, OpenAI, hosting, Backblaze B2, Gmail). **Log the date/time and how it was found.**

### Contain (first hour)
- Stop the bleeding: isolate the affected system (firewall / stop containers); immediately **revoke or rotate any exposed credential**.
- Preserve evidence (logs, a snapshot) before changing things, where feasible.
- If full compromise is suspected, follow **DR Runbook Section C**: rotate ALL secrets and rebuild on a clean server from a **pre-incident** backup.

### Assess — is it a reportable breach?
Decide (a) whether personal information was involved, and (b) whether it poses a **"real risk of significant harm" (RROSH)** — the PIPEDA test. Weigh:
- **Sensitivity** — memorial photos, facial-geometry-derived art, consent records, and contact/order data are sensitive, which **raises** the risk.
- **Probability of misuse** — exposed to a malicious actor? Was it encrypted? Contained quickly?
- **Significant harm** includes identity theft, fraud, humiliation, reputational/relationship damage, and — in our grief context — distress to families.

**Record the assessment and the decision, with reasons.**

### Notify (if RROSH) — do all that apply, **as soon as feasible**
- **Affected individuals** — notify directly (email): what happened, what information was involved, what we've done, what they can do, and how to reach the Privacy Officer.
- **Regulators:**
  - **PIPEDA** — report to the **Office of the Privacy Commissioner of Canada (OPC)** via its breach-report form.
  - **Quebec Law 25** — notify the **Commission d'accès à l'information (CAI)** and affected Quebec residents where there is a risk of serious injury.
  - **GDPR** (if EU/UK individuals affected) — notify the lead supervisory authority **within 72 hours** of becoming aware, where the breach risks people's rights and freedoms.
- **Others who can reduce harm** — e.g., a payment processor or law enforcement.
- **Record-keeping** — log **every** breach (reportable or not) in the breach register (§4) and keep records **≥ 24 months** (PIPEDA), available to the OPC on request.

### Recover
Restore service per the **DR Runbook**; verify integrity; confirm the vulnerability is closed; reconcile orders against Stripe and Printful.

### Review (within 1–2 weeks)
Root-cause analysis; decide what to change (controls, monitoring, process); update this plan and the DR plan; close the register entry.

## 3. Sub-processor breaches
Our processors — **Stripe, Printful, OpenAI, hosting (IONOS/VPS), Backblaze B2, Gmail/SMTP** — are expected to notify us of breaches affecting our data. On notice, treat it as an incident above, assess RROSH for our users, and carry our own notification obligations.

## 4. Breach register (keep ≥ 24 months)
One row per incident: **date detected · description · personal info involved · # individuals · assessment (RROSH? reasons) · actions taken · notifications made (who/when) · status.** A simple spreadsheet suffices; keep it with the ops records.

## 5. Key contacts
- **Privacy Officer / Incident Lead:** the founder — `support@typortrait.com`
- **OPC (Canada):** priv.gc.ca — PIPEDA breach reporting
- **CAI (Quebec):** cai.gouv.qc.ca
- **Sub-processor security/abuse:** Stripe, Printful, OpenAI, hosting, Backblaze
- **Privacy counsel:** *(fill in)*

## 6. Notice to individuals — template
> **Subject:** An important notice about your information
>
> We're writing to let you know about a security incident that may have involved your information.
>
> **What happened:** *(brief factual description + date)*
> **What information was involved:** *(e.g., your email address and the photo you uploaded)*
> **What we've done:** *(containment + fix)*
> **What you can do:** *(specific steps — e.g., be alert to suspicious emails; reset X)*
> **More help:** contact `support@typortrait.com`. You may also contact the Office of the Privacy Commissioner of Canada.

---
*Pairs with: the Disaster Recovery Plan (rebuild/restore) and `ops/RECOVERY-RUNBOOK.md` (exact recovery commands, incl. Section C — breach rebuild).*
