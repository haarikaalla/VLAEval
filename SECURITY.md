# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, email `security@vla-eval.example.com` with:

- A description of the vulnerability and its impact.
- Steps to reproduce (proof-of-concept if possible).
- Any suggested remediation.

We aim to acknowledge reports within 3 business days and provide a resolution
timeline within 10 business days.

## Security Practices in this Repository

- Dependencies are scanned via `pip-audit`/Dependabot and `npm audit` in CI.
- Static analysis via `bandit` (Python) and `eslint-plugin-security` (frontend).
- Container images are scanned with Trivy in the `security-scan` workflow.
- Secrets are never committed; `.env` files are gitignored and `.env.example`
  contains placeholders only.
- API authentication uses signed JWTs and API keys with hashed storage
  (`passlib[bcrypt]`); no plaintext secrets are logged.
- All user input to the FastAPI service is validated via Pydantic schemas.
- CORS is restricted via configuration (`API_CORS_ORIGINS`), not wildcarded in production.
- Docker images run as non-root users.
