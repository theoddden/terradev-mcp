# Quickstart: Publish Terradev MCP Server to the MCP Registry

**Note**: The MCP Registry is currently in preview. Breaking changes or data resets may occur before general availability. If you encounter issues, please report them on [GitHub](https://github.com/modelcontextprotocol/registry/issues).

This tutorial will show you how to publish the Terradev MCP server to the MCP Registry using the official `mcp-publisher` CLI tool.

## Prerequisites

- **Node.js** - Required for the mcp-publisher tool
- **npm account** - The MCP Registry only hosts metadata, not artifacts. We'll publish the package to npm first
- **GitHub account** - We'll use GitHub-based authentication for simplicity

## Step 1: Update package.json for MCP Registry

First, ensure your `package.json` has the required `mcpName` property:

```json
{
  "name": "@theoddden/terradev-mcp",
  "version": "2.0.5",
  "mcpName": "io.github.theoddden/terradev",
  "description": "Complete Agentic GPU Infrastructure for Claude Code — 192 MCP tools: Full training lifecycle, inference deployment with cost guardrails, Ray cluster management, and more",
  "main": "terradev_mcp.py",
  "bin": {
    "terradev-mcp": "terradev_mcp.py"
  },
  "repository": {
    "type": "git",
    "url": "https://github.com/theoddden/terradev-mcp.git"
  },
  "keywords": [
    "gpu", "cloud", "mcp", "claude", "ai", "ml", "kubernetes", 
    "inference", "terraform", "multi-cloud", "numa", "disaggregated-inference"
  ],
  "author": "Terradev",
  "license": "MIT"
}
```

**Important**: The `mcpName` value will be your server's name in the MCP Registry. Since we're using GitHub authentication, it must start with `io.github.theoddden/`.

## Step 2: Build and Publish to npm

The MCP Registry only hosts metadata, so we must first publish the package to npm.

```bash
# Navigate to terradev-mcp directory
cd /Users/theowolfenden/CascadeProjects/terradev-mcp

# Install dependencies
npm install

# If necessary, authenticate to npm
npm adduser

# Publish the package (use --access public for scoped packages)
npm publish --access public
```

Verify your package is published at: https://www.npmjs.com/package/@theoddden/terradev-mcp

## Step 3: Install mcp-publisher

Install the MCP publisher CLI tool:

```bash
# macOS/Linux
curl -L "https://github.com/modelcontextprotocol/registry/releases/latest/download/mcp-publisher_$(uname -s | tr '[:upper:]' '[:lower:]')_$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/').tar.gz" | tar xz mcp-publisher && sudo mv mcp-publisher /usr/local/bin/

# Windows (PowerShell)
$arch = if ([System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -eq "Arm64") { "arm64" } else { "amd64" }; Invoke-WebRequest -Uri "https://github.com/modelcontextprotocol/registry/releases/latest/download/mcp-publisher_windows_$arch.tar.gz" -OutFile "mcp-publisher.tar.gz"; tar xf mcp-publisher.tar.gz mcp-publisher.exe; rm mcp-publisher.tar.gz

# Or with Homebrew (macOS)
brew install mcp-publisher
```

Verify installation:
```bash
mcp-publisher --help
```

## Step 4: Create server.json

Generate the server metadata file:

```bash
mcp-publisher init
```

This creates a `server.json` template. Edit it to match Terradev's configuration:

```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
  "name": "io.github.theoddden/terradev",
  "description": "Complete Agentic GPU Infrastructure for Claude Code — 192 MCP tools for GPU provisioning, training, inference, and multi-cloud management",
  "repository": {
    "url": "https://github.com/theoddden/terradev-mcp",
    "source": "github"
  },
  "version": "2.0.5",
  "packages": [
    {
      "registryType": "npm",
      "identifier": "@theoddden/terradev-mcp",
      "version": "2.0.5",
      "transport": {
        "type": "stdio"
      },
      "environmentVariables": [
        {
          "description": "Terradev API key for GPU cloud providers",
          "isRequired": false,
          "format": "string",
          "isSecret": true,
          "name": "TERRADEV_API_KEY"
        },
        {
          "description": "Path to Terradev credentials file",
          "isRequired": false,
          "format": "string",
          "isSecret": false,
          "name": "TERRADEV_CREDENTIALS_FILE"
        },
        {
          "description": "Default GPU cloud provider",
          "isRequired": false,
          "format": "string",
          "isSecret": false,
          "name": "TERRADEV_PROVIDER"
        }
      ]
    }
  ],
  "categories": [
    "infrastructure",
    "machine-learning",
    "cloud-computing",
    "gpu",
    "automation"
  ],
  "license": "MIT",
  "homepage": "https://github.com/theoddden/terradev-mcp#readme"
}
```

**Key points**:
- `name` must match the `mcpName` from package.json
- Environment variables are optional but recommended for API keys and configuration
- Categories help users discover your server

## Step 5: Authenticate with MCP Registry

Use GitHub authentication (matches your namespace):

```bash
mcp-publisher login github
```

You'll see output like:
```
Logging in with github...

To authenticate, please:
1. Go to: https://github.com/login/device
2. Enter code: ABCD-1234
3. Authorize this application
Waiting for authorization...
```

Visit the GitHub device page, enter the code, and authorize the application. You should see:
```
Successfully authenticated!
✓ Successfully logged in
```

## Step 6: Publish to MCP Registry

Publish your server:

```bash
mcp-publisher publish
```

Expected output:
```
Publishing to https://registry.modelcontextprotocol.io...
✓ Successfully published
✓ Server io.github.theoddden/terradev version 2.0.5
```

## Step 7: Verify Publication

Search for your server in the registry:

```bash
curl "https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.theoddden/terradev"
```

You should see your server metadata in the JSON response.

## Step 8: Update Documentation

Update your README.md to include installation instructions:

```markdown
## Installation via MCP Registry

The Terradev MCP server is published to the MCP Registry. To install it in Claude Desktop:

1. Open Claude Desktop settings
2. Go to "MCP Servers" section
3. Add the server:
   ```json
   {
     "mcpServers": {
       "terradev": {
         "command": "npx",
         "args": ["@theoddden/terradev-mcp"],
         "env": {
           "TERRADEV_PROVIDER": "runpod"
         }
       }
     }
   }
   ```

### Environment Variables (Optional)

- `TERRADEV_API_KEY`: Your Terradev API key
- `TERRADEV_CREDENTIALS_FILE`: Path to credentials file (default: `~/.terradev/credentials.json`)
- `TERRADEV_PROVIDER`: Default cloud provider (e.g., `runpod`, `aws`, `gcp`)
```

## Troubleshooting

| Error Message | Action |
|---------------|--------|
| "Registry validation failed for package" | Ensure `package.json` includes the `mcpName` property matching your server name |
| "Invalid or expired Registry JWT token" | Re-authenticate with `mcp-publisher login github` |
| "You do not have permission to publish this server" | Your GitHub username must match the namespace prefix (`io.github.theoddden/`) |
| "Package not found on npm" | Ensure you published to npm first with `npm publish --access public` |

## Automation with GitHub Actions

To automate publishing on releases, create `.github/workflows/publish-mcp.yml`:

```yaml
name: Publish to MCP Registry

on:
  release:
    types: [published]

jobs:
  publish-mcp:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          
      - name: Install mcp-publisher
        run: |
          curl -L "https://github.com/modelcontextprotocol/registry/releases/latest/download/mcp-publisher_linux_amd64.tar.gz" | tar xz mcp-publisher
          sudo mv mcp-publisher /usr/local/bin/
          
      - name: Authenticate with MCP Registry
        run: mcp-publisher login github
        env:
          MCP_REGISTRY_GITHUB_TOKEN: ${{ secrets.MCP_REGISTRY_GITHUB_TOKEN }}
          
      - name: Publish to MCP Registry
        run: mcp-publisher publish
```

Add `MCP_REGISTRY_GITHUB_TOKEN` as a repository secret with your GitHub personal access token.

## Next Steps

- Monitor your server's usage in the MCP Registry analytics
- Consider adding DNS-based authentication for custom domains
- Explore remote server hosting options for enterprise deployments
- Add semantic versioning and automated testing before publishing

Your Terradev MCP server is now available to all MCP-compatible clients through the registry!
