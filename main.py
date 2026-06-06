from service.rag_service import RAGService
from utils.formatter import format_response, format_user_input


def main():
    print("=" * 50)
    print("金融研报智能问答助手")
    print("=" * 50)
    print("知识库：宁德时代 | 贵州茅台 | 比亚迪 | 新能源行业策略")
    print("输入 'quit' 退出\n")

    service = RAGService()
    print()

    while True:
        user_input = input("你：").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("再见")
            break

        format_user_input(user_input)
        answer, sources = service.chat_with_sources(user_input)
        format_response(answer)

        if sources:
            print("【信息来源】")
            for s in sources:
                print(f"  · {s}")
            print()


if __name__ == "__main__":
    main()
