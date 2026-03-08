#!/bin/bash

echo "🚀 Publishing Terradev MCP Server to Registry"
echo "=========================================="

# Check if we have the mcp-publisher
if [ ! -f "./mcp-publisher" ]; then
    echo "❌ mcp-publisher not found. Downloading..."
    curl -L "https://github.com/modelcontextprotocol/registry/releases/latest/download/mcp-publisher_darwin_amd64.tar.gz" | tar xz mcp-publisher
    chmod +x mcp-publisher
fi

echo "✅ mcp-publisher ready"

# Check package.json has required fields
if ! grep -q "mcpName" package.json; then
    echo "❌ package.json missing mcpName field"
    exit 1
fi

echo "✅ package.json validated"

# Initialize server.json if it doesn't exist
if [ ! -f "server.json" ]; then
    echo "📝 Creating server.json..."
    ./mcp-publisher init
    echo "⚠️  Please edit server.json to match your configuration before publishing"
    echo "   Current contents:"
    cat server.json
    echo ""
    echo "   Press Enter to continue or Ctrl+C to edit..."
    read
fi

echo "✅ server.json ready"

# Login to MCP Registry
echo "🔐 Logging into MCP Registry..."
./mcp-publisher login github

# Publish to MCP Registry
echo "📦 Publishing to MCP Registry..."
./mcp-publisher publish

echo ""
echo "🎉 Publishing complete!"
echo ""
echo "To verify, run:"
echo "curl \"https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.theoddden/terradev\""
