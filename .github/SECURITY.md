# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **DO NOT** open a public issue for security vulnerabilities
2. Email the maintainers directly or use GitHub's private vulnerability reporting
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- Acknowledgment within 48 hours
- Status update within 7 days
- Credit in the security advisory (if desired)

### Scope

Security issues we're interested in:
- Path traversal vulnerabilities
- XML/SSML injection attacks
- Authentication bypass
- Arbitrary code execution
- Denial of service
- Information disclosure

### Out of Scope

- Issues in dependencies (report upstream)
- Issues requiring physical access
- Social engineering attacks

## Security Measures

Voice Soundboard implements:

- **Path Traversal Protection**: All file paths validated with `sanitize_filename()`
- **XXE Protection**: SSML parsed with defusedxml
- **Rate Limiting**: Token bucket algorithm prevents abuse
- **Input Validation**: Length limits and type checking
- **WebSocket Security**: Origin validation, API key auth, TLS support
- **Safe Error Messages**: Internal paths never exposed

See [SECURITY_AUDIT.md](../SECURITY_AUDIT.md) for the full security audit.

## Security Updates

Security updates are released as patch versions (e.g., 0.1.1) and announced via:
- GitHub Security Advisories
- Release notes

We recommend always using the latest version.
