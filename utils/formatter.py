def format_response(text: str) ->str:
    print("\n"+"-"*50)
    print("🤖 AI助手")
    print("-"*50)
    print(text)
    print("-"*50+"\n")
    return text      

def format_user_input(messages :str) ->str:
    print(f'\n👤 你:{messages}')
    return messages
