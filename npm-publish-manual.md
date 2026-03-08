# Manual npm Publishing Instructions

Since Node.js installation is complex in this environment, here are the manual steps to publish to npm:

## 1. Install Node.js (on your local machine)

```bash
# macOS
brew install node

# Or download from https://nodejs.org/
```

## 2. Navigate to terradev-mcp directory

```bash
cd /Users/theowolfenden/CascadeProjects/terradev-mcp
```

## 3. Login to npm

```bash
npm login
# Enter your npm username, password, and email
```

## 4. Verify package.json

```bash
cat package.json | grep '"name"'
# Should show: "@theoddden/terradev-mcp"

cat package.json | grep '"mcpName"'
# Should show: "mcpName": "io.github.theoddden/terradev"
```

## 5. Publish to npm

```bash
npm publish --access public
```

## 6. Verify npm publication

Visit: https://www.npmjs.com/package/@theoddden/terradev-mcp

## 7. Then publish to MCP Registry

```bash
./mcp-publisher publish
```

## 8. Verify MCP Registry

```bash
curl "https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.theoddden/terradev"
```

## Current Status

✅ All files are ready for publication
✅ MCP Registry authentication completed
✅ server.json validated and fixed (description under 100 chars)
❌ npm publishing requires manual Node.js installation

## Next Steps

1. Install Node.js on your local machine
2. Run `npm login` and `npm publish --access public`
3. Run `./mcp-publisher publish` to complete MCP Registry publication

The MCP server is ready to ship once npm publishing is completed!
