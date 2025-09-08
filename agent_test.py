import agent
import os

# Fonction utilitaire pour lire la clé API selon le provider depuis api_conf.conf


if __name__ == "__main__":
    provider = "gemini"  # ou "openai"
    test_agent = agent.TestAgent(provider=provider)
    test_agent.run_all_tests()
