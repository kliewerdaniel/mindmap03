# Security Documentation

This document outlines the security measures and best practices implemented in Mind Map AI to protect user data and ensure safe operation.

## Security Features

### Input Validation

**File Upload Security:**
- Strict file extension validation (`.md`, `.txt`, `.zip` only)
- File size limits (10MB maximum upload size)
- Content size validation on text ingestion

**Text Content Validation:**
- Automatic validation of filename extensions
- Content size validation with configurable limits
- URL and path sanitization where applicable

### Rate Limiting

**API Rate Limiting:**
- 100 requests per minute per IP address
- 1000 requests per hour per IP address
- Configurable rate limits via environment variables

**Rate Limit Headers:**
API responses include rate limiting headers:
- `X-RateLimit-Limit`: Request limit per window
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when the rate limit resets

### Local-First LLM Architecture

**External LLM Protection:**
- Configurable `disable_external_llm` setting
- Local Ollama instance enforcement
- No external API calls by default

**Network Boundaries:**
- All LLM processing happens locally
- Network isolation prevents data leakage
- No telemetry or external logging by default

## Security Configuration

### Environment Variables

```bash
# Security Settings
MAX_UPLOAD_SIZE=10485760          # 10MB in bytes
ALLOWED_EXTENSIONS=".md,.txt,.zip"
DISABLE_EXTERNAL_LLM=false       # Set to true for local-only enforcement

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_REQUESTS_PER_HOUR=1000
```

### Configurable Security Limits

**File Handling:**
- Maximum upload size: 10MB (configurable)
- Allowed file types: Markdown, plain text, and ZIP archives
- Automatic file type detection and validation

**API Security:**
- CORS configuration with explicit origins
- Request size limits on all endpoints
- Input sanitization on all text inputs

## Safe File Handling

### Upload Processing
1. **Extension Check**: Validate file extension against whitelist
2. **Size Check**: Verify file size doesn't exceed limits
3. **Content Validation**: Scan content for malicious patterns
4. **Secure Storage**: Store files with proper permissions

### ZIP Archive Handling
- Validate ZIP structure before processing
- Extract only allowed file types from archives
- Prevent directory traversal attacks
- Clean up temporary files after processing

## Best Practices for Deployment

### Network Security
- Run behind reverse proxy (nginx/apache) in production
- Use HTTPS/TLS encryption
- Restrict CORS origins to specific domains
- Implement proper firewall rules

### Container Security (Docker)
- Use non-root user for application container
- Run with read-only filesystem where possible
- Implement proper secrets management
- Scan images for vulnerabilities

### API Security
- Implement proper authentication if needed
- Use API keys for production deployments
- Monitor API usage and anomalies
- Implement proper logging and alerting

## Monitoring and Alerting

### Security Monitoring
- Log all validation failures
- Monitor rate limiting events
- Alert on suspicious activity patterns
- Track API usage by endpoint and IP

### Audit Logging
- Log all file upload attempts
- Record validation failures
- Track processing events
- Maintain audit trail for compliance

## Incident Response

### Security Incidents
1. **Detection**: Monitor logs and alerts
2. **Assessment**: Evaluate incident severity
3. **Containment**: Disable affected features
4. **Recovery**: Restore from clean backups
5. **Analysis**: Review root cause and implement fixes

### Reporting
- Document all security incidents
- Update security policies as needed
- Improve monitoring and detection
- Share lessons learned with the community

## Compliance Considerations

### Data Privacy
- Process all data locally by default
- Support GDPR compliance through local processing
- No data shared with external services
- User full control over data retention

### SOX and Enterprise Compliance
- Detailed audit logging capabilities
- Configurable security policies
- Role-based access control ready architecture
- Data encryption at rest options

## Development Security

### Code Security
- Input validation on all user inputs
- Proper error handling without information disclosure
- Secure defaults for all configuration
- Regular security dependency updates

### Testing
- Security unit tests for validation logic
- Integration tests for rate limiting
- Penetration testing guidelines
- Continuous security scanning in CI/CD

## Future Security Enhancements

### Planned Features
- OAuth2 authentication integration
- API key management system
- Advanced rate limiting with Redis backend
- File scanning for malware (optional)
- End-to-end encryption for sensitive data

### Security Roadmap
- Implement securityheaders.com best practices
- Add Content Security Policy headers
- Implement proper session management
- Add database encryption options

This security documentation ensures Mind Map AI maintains a strong security posture while respecting user privacy and data sovereignty through its local-first architecture.
