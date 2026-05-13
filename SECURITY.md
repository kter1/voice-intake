# Security Policy

## Status

This repository is a portfolio / reference implementation that runs against
synthetic demo data. It is not a production clinical system, a HIPAA-compliant
deployment, a payer or EHR-integrated product, a regulatory-cleared service,
or a source of medical advice.

The code is provided to demonstrate architecture patterns. Anyone running it
is responsible for their own security posture, provider terms, and data
handling.

## Reporting a Vulnerability

Please do not open a public issue or pull request that contains exploit
details, credentials, or steps that could compromise others.

Contact: Contact the repository owner through GitHub. Do not include exploit
details in a public issue.

When reporting, please include:

- The affected component or file path
- A minimal description of the issue
- Steps to reproduce (kept private)
- Potential impact
- Suggested remediation, if known

A reasonable best-effort response is intended, but this project comes with no
service-level guarantee and no formal vulnerability-response program.

## Sensitive Data

Do not submit any of the following to this repository, its issue tracker, or
any associated discussion:

- Real patient data or Protected Health Information (PHI)
- Real names, dates of birth, addresses, or contact details for real people
- Production call recordings, transcripts, or session logs
- API keys, tokens, credentials, or environment files
- Confidential operational data from any organization

All seeded RAG content, fixtures, and demo flows in this repository use
synthetic data only.

## Scope

This security policy covers the source code in this repository. It does not
cover:

- Third-party providers configured via `.env` (LLM provider, Deepgram, Twilio, etc.)
- Local environments or operator infrastructure
- Forks or derivative deployments

For issues in third-party services, please report directly to the relevant
vendor.
