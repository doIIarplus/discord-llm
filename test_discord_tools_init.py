#!/usr/bin/env python3
"""Test script for Discord tools initialization"""

import sys
import os
import asyncio

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.base import registry
# Importing this registers the tools
import tools.discord_tools

def test_discord_tools_init():
    """Test that all Discord tools are registered correctly"""
    print("Testing Discord tools registration...")
    
    # List of expected tool names
    expected_tools = [
        "get_channel_messages",
        "get_user_info", 
        "get_channel_info",
        "list_channels",
        "get_server_info",
        "search_messages",
        "get_online_members",
        "delete_message",
        "purge_messages",
        "list_roles",
        "add_role",
        "remove_role",
        "create_channel",
        "set_channel_topic",
        "pin_message",
        "create_invite"
    ]
    
    registered_tools = [t.name for t in registry.get_all()]
    
    missing_tools = []
    for tool_name in expected_tools:
        if tool_name not in registered_tools:
            missing_tools.append(tool_name)
    
    if missing_tools:
        print(f"FAILED: Missing tools: {missing_tools}")
        return False
        
    print(f"SUCCESS: All {len(expected_tools)} expected tools are registered.")
    
    # Check specific tool schemas to verify parameters
    print("\nVerifying schemas...")
    
    # Check delete_message
    delete_tool = registry.get("delete_message")
    params = {p.name: p for p in delete_tool.parameters}
    if "message_id" not in params or "channel_id" not in params or "reason" not in params:
        print("FAILED: delete_message tool missing parameters")
        return False
    
    # Check create_channel
    create_tool = registry.get("create_channel")
    params = {p.name: p for p in create_tool.parameters}
    if "name" not in params or "type" not in params or "category_name" not in params:
        print("FAILED: create_channel tool missing parameters")
        return False
        
    print("SUCCESS: Tool schemas verify correctly.")
    return True

if __name__ == "__main__":
    success = test_discord_tools_init()
    sys.exit(0 if success else 1)
