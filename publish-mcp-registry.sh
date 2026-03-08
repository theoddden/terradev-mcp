#!/bin/bash

echo "🚀 Terradev MCP Server - MCP Registry Publishing"
echo "=============================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check files
echo -e "${GREEN}[1/6]${NC} Checking configuration files..."

if [ ! -f "package.json" ]; then
    echo -e "${RED}❌ package.json not found${NC}"
    exit 1
fi

if [ ! -f "server.json" ]; then
    echo -e "${RED}❌ server.json not found${NC}"
    exit 1
fi

# Validate package.json has mcpName
if ! grep -q "mcpName" package.json; then
    echo -e "${RED}❌ package.json missing mcpName field${NC}"
    exit 1
fi

# Validate names match
MCP_NAME_PACKAGE=$(grep -o '"mcpName": *"[^"]*"' package.json | cut -d'"' -f4)
MCP_NAME_SERVER=$(grep -o '"name": *"[^"]*"' server.json | head -1 | cut -d'"' -f4)

if [ "$MCP_NAME_PACKAGE" != "$MCP_NAME_SERVER" ]; then
    echo -e "${RED}❌ Name mismatch: package.json ($MCP_NAME_PACKAGE) != server.json ($MCP_NAME_SERVER)${NC}"
    exit 1
fi

echo -e "${GREEN}   ✅ Names match: $MCP_NAME${NC}"

# Check mcp-publisher exists
if [ ! -f "./mcp-publisher" ]; then
    echo -e "${YELLOW}[2/6]${NC} Downloading mcp-publisher..."
    curl -L "https://github.com/modelcontextprotocol/registry/releases/latest/download/mcp-publisher_darwin_amd64.tar.gz" | tar xz mcp-publisher 2>/dev/null || curl -L "https://github.com/modelcontextprotocol/registry/releases/latest/download/mcp-publisher_linux_amd64.tar.gz" | tar xz mcp-publisher
    chmod +x mcp-publisher
fi

echo -e "${GREEN}[3/6]${NC} mcp-publisher ready"

# Show current configuration
echo -e "${GREEN}[4/6]${NC} Current configuration:"
echo "   Package: $(grep -o '"name": *"[^"]*"' package.json | head -1 | cut -d'"' -f4)"
echo "   Version: $(grep -o '"version": *"[^"]*"' package.json | cut -d'"' -f4)"
echo "   MCP Name: $MCP_NAME"
echo ""

# Authenticate
echo -e "${GREEN}[5/6]${NC} Authenticating with MCP Registry..."
./mcp-publisher login github

# Publish
echo -e "${GREEN}[6/6]${NC} Publishing to MCP Registry..."
if ./mcp-publisher publish; then
    echo ""
    echo -e "${GREEN}🎉 Successfully published to MCP Registry!${NC}"
    echo ""
    echo "📦 Package: https://www.npmjs.com/package/@theoddden/terradev-mcp"
    echo "🔍 Registry: https://registry.modelcontextprotocol.io/servers"
    echo ""
    echo "🧪 Verify with:"
    echo "curl \"https://registry.modelcontextprotocol.io/v0.1/servers?search=$MCP_NAME\""
    echo ""
    echo "📋 Claude Desktop Installation:"
    echo "{"
    echo "  \"mcpServers\": {"
    echo "    \"terradev\": {"
    echo "      \"command\": \"npx\","
    echo "      \"args\": [\"@theoddden/terradev-mcp\"],"
    echo "      \"env\": {"
    echo "        \"TERRADEV_PROVIDER\": \"runpod\""
    echo "      }"
    echo "    }"
    echo "  }"
    echo "}"
else
    echo -e "${RED}❌ Publishing failed${NC}"
    echo "Check the error message above and ensure:"
    echo "1. You're logged into npm (npm publish must be done first)"
    echo "2. The package exists on npm"
    echo "3. Your GitHub username matches the namespace"
    exit 1
fi
