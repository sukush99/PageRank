from config import config

def main():
    Link = 'https://en.wikipedia.org/wiki/Python_(programming_language)'
    print(f"Link: {Link}")
    print(f"Link from config: {config.link}")


if __name__ == "__main__":
    main()
