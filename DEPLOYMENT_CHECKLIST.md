# Terradev MCP Server Deployment Checklist

## ✅ Pre-Deployment Checklist

### 1. Package Configuration
- [x] `package.json` updated with `mcpName: "io.github.theoddden/terradev"`
- [x] Package name changed to `@theoddden/terradev-mcp` (scoped package)
- [x] Repository URL correctly pointing to GitHub
- [x] Version set to `2.0.5`

### 2. Server Metadata
- [x] `server.json` created with proper schema
- [x] Server name matches `mcpName` from package.json
- [x] Environment variables documented (API key, credentials file, provider)
- [x] Categories added for discoverability
- [x] License and homepage set correctly

### 3. Tools Ready
- [x] `mcp-publisher` binary downloaded and executable
- [x] `publish.sh` script created for automated publishing

## 🚀 Deployment Steps

### Step 1: Publish to npm (Manual)
```bash
# This requires npm to be installed and npm login
npm login
npm publish --access public
```

### Step 2: Authenticate with MCP Registry
```bash
./mcp-publisher login github
```

### Step 3: Publish to MCP Registry
```bash
./mcp-publisher publish
```

### Step 4: Verify Publication
```bash
curl "https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.theoddden/terradev"
```

## 📋 Post-Deployment

### Update Documentation
- [ ] Update README.md with MCP Registry installation instructions
- [ ] Add "Installation via MCP Registry" section
- [ ] Update Claude Desktop configuration examples

### Create GitHub Release
- [ ] Tag release: `git tag v2.0.5`
- [ ] Push tag: `git push origin v2.0.5`
- [ ] Create GitHub release with changelog

### Monitor Adoption
- [ ] Check registry analytics (when available)
- [ ] Monitor GitHub stars and issues
- [ ] Track npm downloads

## 🔧 Troubleshooting

### Common Issues
1. **"Registry validation failed"** - Ensure `mcpName` matches between package.json and server.json
2. **"Permission denied"** - GitHub username must match namespace prefix
3. **"Package not found"** - Publish to npm first before MCP Registry

### Recovery Commands
```bash
# Re-authenticate if token expires
./mcp-publisher login github

# Re-initialize server.json if corrupted
rm server.json
./mcp-publisher init

# Check current configuration
cat package.json | grep mcpName
cat server.json | grep name
```

## 📊 Expected Results

After successful deployment:
- Package available at: https://www.npmjs.com/package/@theoddden/terradev-mcp
- Server listed in MCP Registry under: `io.github.theoddden/terradev`
- Installation via: `npx @theoddden/terradev-mcp`
- Claude Desktop configuration ready

## 🎯 Success Metrics

- [ ] MCP Registry returns server metadata
- [ ] npm package page shows correct version
- [ ] Claude Desktop can install and run server
- [ ] All 192 MCP tools are discoverable
