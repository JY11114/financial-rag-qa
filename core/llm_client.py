from anthropic import Anthropic
from core.config import Config

class LLMClient:
    def __init__(self):
        self.client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    
    def send_message(self,messages : list,system : str= None) ->str:
        """
        发送消息给Claude,并返回回复的文字内容
        
        参数:
            messages: 对话历史列表,格式是[{"role":"user","content","..."},...]
            system:   系统提示词,可以不传(None表示不设置)
        返回:
            str: Claude回复的文字内容    
            """    
        kwargs = {
            "model":Config.MODEL_NAME,
            "max_tokens":Config.MAX_TOKENS,
            "messages":messages,
        }

        if system is not None:
            kwargs["system"]= system

        response = self.client.messages.create(**kwargs)

        return response.content[0].text
    



