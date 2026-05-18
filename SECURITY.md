# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of Metadata-to-Morphsource-Compare seriously. If you discover a security vulnerability, please follow these steps:

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **Preferred**: Open a security advisory on GitHub
   - Go to the [Security tab](https://github.com/johntrue15/MorphoClaw/security)
   - Click "Report a vulnerability"
   - Provide detailed information about the vulnerability

2. **Alternative**: Email the maintainers directly
   - Check the repository for maintainer contact information
   - Include "SECURITY" in the subject line
   - Provide detailed information about the vulnerability

### What to Include

When reporting a vulnerability, please include:

- **Description**: A clear description of the vulnerability
- **Impact**: What kind of data or systems could be affected
- **Reproduction**: Step-by-step instructions to reproduce the issue
- **Proof of Concept**: Code, screenshots, or examples demonstrating the vulnerability
- **Suggested Fix**: If you have ideas on how to fix it (optional)
- **Your Contact Information**: So we can follow up with questions

### What to Expect

- **Acknowledgment**: We'll acknowledge receipt of your report within 48 hours
- **Initial Assessment**: We'll provide an initial assessment within 5 business days
- **Updates**: We'll keep you informed about our progress fixing the vulnerability
- **Fix Timeline**: We aim to release fixes for critical vulnerabilities within 30 days
- **Credit**: We'll give you credit for the discovery (unless you prefer to remain anonymous)

## Security Best Practices

### API Keys and Secrets

- **Never commit API keys** to version control
- Use the `.env` file for local development (already in `.gitignore`)
- Use GitHub Secrets for CI/CD workflows
- Rotate API keys regularly
- Use minimum necessary permissions for API keys

### Environment Variables

- Copy `.env.example` to `.env` for local development
- Never share your `.env` file
- Keep sensitive credentials out of logs and error messages

### Dependencies

- Keep dependencies up to date
- Review security advisories for dependencies
- Use `pip install --upgrade` regularly
- Consider using tools like `safety` to check for known vulnerabilities:
  ```bash
  pip install safety
  safety check
  ```

### Data Handling

- Be cautious with specimen data that may be sensitive
- Follow your institution's data handling policies
- Don't upload private data to public repositories
- Use appropriate access controls for MorphoSource API

## Disclosure Policy

- We will coordinate the disclosure timeline with you
- We prefer coordinated disclosure after a fix is available
- We'll credit security researchers (unless they prefer anonymity)
- We'll publish security advisories for confirmed vulnerabilities

## Security Updates

Security updates will be:
- Released as soon as possible after verification
- Announced in GitHub Security Advisories
- Documented in release notes
- Communicated to users through GitHub notifications

## Scope

### In Scope

- Authentication and authorization issues
- Data exposure or leakage
- Injection vulnerabilities (SQL, command, etc.)
- API security issues
- Sensitive data in logs or error messages
- Dependency vulnerabilities

### Out of Scope

- Issues in external services (MorphoSource, OpenAI)
- Social engineering attacks
- Denial of service attacks
- Issues requiring physical access
- Issues in development/test environments

## Contact

For security-related questions that are not vulnerabilities, please open a regular GitHub issue or discussion.

Thank you for helping keep Metadata-to-Morphsource-Compare and our users safe!
