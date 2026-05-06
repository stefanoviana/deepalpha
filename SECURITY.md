# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

- **Do NOT** open a public issue
- Email: **hello@deepalphabot.com**
- Include: description, steps to reproduce, potential impact
- We will respond within 48 hours

## Supported Versions

| Version | Supported |
|---------|-----------|
| v11.x   | ✅ Active |
| v10.x   | ⚠️ Security fixes only |
| < v10   | ❌ Not supported |

## Security Measures

- API keys encrypted with AES-256 Fernet
- Non-custodial: bot can trade but never withdraw
- JWT authentication with 7-day expiry
- Rate limiting on all endpoints
- HMAC signature verification on webhooks
- No credentials stored in source code
