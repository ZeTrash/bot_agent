import sqlite3
from collections import defaultdict, deque
from typing import List, Optional
import openai
import requests  # Ajouté pour Gemini
import time
import os
from dotenv import load_dotenv

load_dotenv()

# --- Classe Memory avec SQLite et trim automatique ---
class Memory:
    """
    Classe Memory : gère la mémoire courte (volatile) et longue (persistante via SQLite) de l'agent conversationnel.
    - Mémoire courte : stocke les derniers messages pour le contexte immédiat (FIFO, trim par taille/token), isolée par utilisateur.
    - Mémoire longue : stocke des informations importantes de façon persistante (clé/valeur), isolée par utilisateur.
    Utilisation :
        memory = Memory(short_size=10, db_path="agent_memory.db")
        memory.add_short(user_id, "message")
        memory.add_long(user_id, "clé", "valeur")
    """
    def __init__(self, short_size: int = 10, db_path: str = "agent_memory.db", max_tokens_short: int = 2000):
        self.short_size = short_size
        self.short_context = defaultdict(lambda: deque(maxlen=short_size))  # Isolation par user_id
        self.db_path = db_path
        self.max_tokens_short = max_tokens_short
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS long_memory (
                user_id TEXT,
                key TEXT,
                value TEXT,
                PRIMARY KEY (user_id, key)
            )
        """)
        conn.commit()
        conn.close()

    def add_short(self, user_id: str, message: str):
        self.short_context[user_id].append(message)
        self.trim_short(user_id)

    def get_short(self, user_id: str) -> List[str]:
        return list(self.short_context[user_id])

    def reset_short(self, user_id: str):
        self.short_context[user_id].clear()

    def trim_short(self, user_id: str):
        while sum(len(m.split()) for m in self.short_context[user_id]) > self.max_tokens_short:
            self.short_context[user_id].popleft()

    def add_long(self, user_id: str, key: str, value: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO long_memory (user_id, key, value) VALUES (?, ?, ?)", (user_id, key, value))
        conn.commit()
        conn.close()

    def get_long(self, user_id: str, key: str) -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT value FROM long_memory WHERE user_id = ? AND key = ?", (user_id, key))
        row = c.fetchone()
        conn.close()
        return row[0] if row else None

    def get_all_long(self, user_id: str) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT value FROM long_memory WHERE user_id = ?", (user_id,))
        rows = c.fetchall()
        conn.close()
        return [row[0] for row in rows]

# --- Classe Agent AI avancée ---
class Agent:
    """
    Classe Agent : agent conversationnel intelligent, multi-provider (OpenAI, Gemini).
    - Gère la mémoire (Memory)
    - Construit les prompts avec contexte
    - Sélectionne dynamiquement le modèle selon le provider
    - Route les requêtes vers OpenAI ou Gemini selon le provider
    Utilisation :
        agent = Agent(api_key, provider="openai" ou "gemini")
        response = agent.send_prompt(user_id, message, commands)
    """
    MODEL_PRIORITY = {
        "openai": [
            "gpt-5", "gpt-5-mini",
            "gpt-4.1", "gpt-4.1-mini",
            "gpt-4o", "gpt-4o-mini",
            "gpt-3.5-turbo"
        ],
        "gemini": [
            "gemini-1.5-flash", "gemini-1.5-pro",
            "gemini-2.0-flash", "gemini-2.0-pro",
            "gemini-2.5-flash", "gemini-2.5-pro",
        ]
    }
    DEFAULT_MODEL = {
        "openai": "gpt-4o",
        "gemini": "gemini-2.5-pro"
    }

    def __init__(self, default_model: str = None, memory_config: dict = None, provider: str = "openai"):
        """
        Initialise l'agent conversationnel.
        :param api_key: Clé API (OpenAI ou Gemini)
        :param default_model: Modèle à utiliser (optionnel, sinon choisi dynamiquement)
        :param memory_config: Configuration de la mémoire (dictionnaire)
        :param provider: Provider à utiliser ("openai" ou "gemini")
        """
        self.provider = provider
        self.api_key = self.get_api_key()
        self.model = default_model or self.DEFAULT_MODEL.get(provider, "gpt-4o")
        self.memory = Memory(**(memory_config or {}))
    
    def get_api_key(self, conf_path=".env"):
        # Lecture d'abord via variables d'environnement
        env_key = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
        }[self.provider]
        if env_key:
            return env_key
        # Fallback ancien fichier de conf
        key_name = {
            "openai": "openai_api_key",
            "gemini": "gemini_api_key"
        }[self.provider]
        try:
            with open(conf_path, "r") as f:
                for line in f:
                    if line.startswith(key_name):
                        return line.split("=", 1)[1].strip()
        except FileNotFoundError:
            pass
        raise ValueError(f"Clé API pour {self.provider} non trouvée dans .env ni dans {conf_path}.")

    def set_api_key(self, conf_path=".env", new_key: str = None):
        # Méthode legacy conservée, mais on encourage l'usage des variables d'environnement
        key_name = {
            self.provider: self.provider+"_api_key",
        }[self.provider]
        with open(conf_path, "a") as f:
            f.write(f"{key_name}={new_key}\n")
            self.api_key = new_key
            self.model = None

    def update_api_key(self, new_key: str):
        """
        Met à jour la clé API de l'agent.
        :param new_key: Nouvelle clé API
        """
        self.api_key = new_key

    def select_model(self, commands: Optional[List[str]] = None) -> str:
        """
        Sélectionne dynamiquement le modèle à utiliser selon le provider et les commandes.
        :param commands: Liste de commandes optionnelles
        :return: Nom du modèle
        """
        if self.provider == "gemini":
            return self.model or self.DEFAULT_MODEL["gemini"]
        if commands:
            if any(cmd in ["summarize", "translate"] for cmd in commands):
                return "gpt-4o-mini"
        return self.model or self.DEFAULT_MODEL["openai"]

    def build_prompt(self, user_id: str, message: str, commands: Optional[List[str]] = None) -> str:
        """
        Construit le prompt complet envoyé au LLM, incluant le contexte mémoire et les commandes.
        :param user_id: Identifiant utilisateur
        :param message: Message utilisateur
        :param commands: Liste de commandes optionnelles
        :return: Prompt complet (str)
        """
        system_prompt = "Vous êtes un assistant intelligent et utile."
        short_ctx = "\n".join(self.memory.get_short(user_id))
        long_ctx = "\n".join(self.memory.get_all_long(user_id))
        cmd_prompt = f"Commandes : {commands}" if commands else ""
        prompt = f"{system_prompt}\nMémoire courte:\n{short_ctx}\nMémoire longue:\n{long_ctx}\n{cmd_prompt}\nUtilisateur({user_id}): {message}"
        return prompt

    def post_process(self, response: str, commands: Optional[List[str]] = None) -> dict:
        """
        Post-traite la réponse brute du LLM pour ajouter des champs (résumé, sentiment, etc.).
        :param response: Réponse brute du LLM
        :param commands: Liste de commandes optionnelles
        :return: Dictionnaire enrichi
        """
        result = {"content": response}
        if commands:
            if "summarize" in commands:
                result["summary"] = response[:100] + "..."
            if "sentiment" in commands:
                result["sentiment"] = {"score": 0.0, "label": "neutral"}
        return result

    def send_prompt(self, user_id: str, message: str, commands: Optional[List[str]] = None) -> dict:
        """
        Envoie un prompt à l'API LLM (OpenAI ou Gemini) et gère la mémoire.
        :param user_id: Identifiant utilisateur
        :param message: Message utilisateur
        :param commands: Liste de commandes optionnelles
        :return: Dictionnaire réponse post-traitée
        """
        prompt = self.build_prompt(user_id, message, commands)
        selected_model = self.select_model(commands)

        if self.provider == "openai":
            openai.api_key = self.api_key
            try:
                resp = openai.chat.completions.create(
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=500
                )
                response_text = resp.choices[0].message.content
            except Exception as e:
                response_text = f"Erreur lors de l'appel OpenAI : {str(e)}"
        elif self.provider == "gemini":
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent"
                headers = {"Content-Type": "application/json"}
                params = {"key": self.api_key}
                data = {
                    "contents": [
                        {"parts": [{"text": prompt}]}
                    ]
                }
                r = requests.post(url, headers=headers, params=params, json=data)
                r.raise_for_status()
                result = r.json()
                response_text = result["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                response_text = f"Erreur lors de l'appel Gemini : {str(e)}"
        else:
            response_text = "Provider non supporté."

        # Mise à jour mémoire isolée par utilisateur
        self.memory.add_short(user_id, f"Utilisateur({user_id}): {message}")
        self.memory.add_short(user_id, f"Agent: {response_text}")
        self.memory.add_long(user_id, f"{user_id}_{len(self.memory.get_short(user_id))}", message)

        return self.post_process(response_text, commands)

# --- Classes de test ---
class TestClient:
    """
    TestClient permet de simuler un client pour tester les interactions avec la classe Agent.
    Utilisation typique :
        - Instancier avec la clé API, le provider ("openai" ou "gemini") et éventuellement une config mémoire.
        - Utiliser la méthode send pour envoyer un message et afficher la réponse.
    """
    def __init__(self, memory_config: dict = None, provider: str = "openai", default_model: str = None):
        """
        Initialise un TestClient.
        :param api_key: Clé API à utiliser (OpenAI ou Gemini)
        :param memory_config: Configuration de la mémoire (dictionnaire)
        :param provider: Provider à utiliser ("openai" ou "gemini")
        :param default_model: Modèle à utiliser (optionnel, sinon choisi dynamiquement)
        """
        self.agent = Agent(api_key, default_model=default_model, memory_config=memory_config, provider=provider)

    def send(self, user_id: str, message: str, commands: Optional[List[str]] = None):
        """
        Envoie un message à l'agent et affiche la réponse.
        :param user_id: Identifiant utilisateur
        :param message: Message à envoyer
        :param commands: Liste de commandes optionnelles (ex: ["summarize"])
        :return: Réponse de l'agent (dict)
        """
        response = self.agent.send_prompt(user_id, message, commands)
        print(f"[TestClient] Réponse: {response}")
        return response

class TestAgent:
    """
    TestAgent permet de réaliser des tests structurés et avancés sur la classe Agent.
    Il propose plusieurs scénarios de test :
        - test_message_simple : test d'un message basique
        - test_message_avec_commandes : test avec commande (ex: summarize)
        - test_memoire : test avancé de la mémoire (remplissage, trim, contexte)
    Utilisation typique :
        - Instancier avec la clé API, le provider ("openai" ou "gemini") et éventuellement une config mémoire.
        - Appeler run_all_tests() pour exécuter tous les scénarios.
    """
    def __init__(self, memory_config: dict = None, provider: str = "openai", default_model: str = None):
        """
        Initialise un TestAgent.
        :param api_key: Clé API à utiliser (OpenAI ou Gemini)
        :param memory_config: Configuration de la mémoire (dictionnaire)
        :param provider: Provider à utiliser ("openai" ou "gemini")
        :param default_model: Modèle à utiliser (optionnel, sinon choisi dynamiquement)
        """
        self.agent = Agent( default_model=default_model, memory_config=memory_config, provider=provider)

    def test_message_simple(self):
        """
        Teste l'envoi d'un message simple à l'agent et affiche la réponse.
        """
        user_id = "test_user"
        message = "Bonjour, que peux-tu faire ?"
        print("[TestAgent] Test message simple...")
        response = self.agent.send_prompt(user_id, message)
        print(f"[TestAgent] Réponse: {response}")
        return response

    def test_message_avec_commandes(self):
        """
        Teste l'envoi d'un message avec une commande (ex: summarize) et affiche la réponse.
        """
        user_id = "test_user"
        message = "Peux-tu résumer ce texte ?"
        commands = ["summarize"]
        print("[TestAgent] Test message avec commande 'summarize'...")
        response = self.agent.send_prompt(user_id, message, commands)
        print(f"[TestAgent] Réponse: {response}")
        return response

    def test_memoire(self):
        """
        Teste la gestion de la mémoire de l'agent :
            - Envoie plusieurs messages pour remplir la mémoire courte
            - Affiche la mémoire courte après chaque message
            - Affiche la mémoire longue
            - Vérifie l'utilisation du contexte dans une nouvelle requête
        """
        user_id = "memoire_user"
        print("[TestAgent] Test mémoire avancée...")
        messages = [
            "Premier message de test.",
            "Deuxième message pour la mémoire.",
            "Troisième message, on continue.",
            "Quatrième message, la mémoire courte va-t-elle suivre ?",
            "Cinquième message, testons le trim !"
        ]
        for i, msg in enumerate(messages):
            print(f"\n[Envoi {i+1}] {msg}")
            self.agent.send_prompt(user_id, msg)
            print(f"Mémoire courte après {i+1} messages : {self.agent.memory.get_short(user_id)}")
        print("\nMémoire longue :", self.agent.memory.get_all_long(user_id))
        # Vérification du contexte dans une nouvelle requête
        print("\n[Test du contexte dans une nouvelle requête]")
        response = self.agent.send_prompt(user_id, "Que sais-tu de notre conversation ?")
        print(f"Réponse avec contexte : {response}")
        return response

    def test_isolation_utilisateur(self):
        """
        Teste que la mémoire courte et longue est bien isolée entre deux utilisateurs différents.
        """
        user_a = "user_A"
        user_b = "user_B"
        print("[TestAgent] Test isolation mémoire entre utilisateurs...")
        self.agent.send_prompt(user_a, "Message A1")
        self.agent.send_prompt(user_a, "Message A2")
        self.agent.send_prompt(user_b, "Message B1")
        print(f"Mémoire courte user_A : {self.agent.memory.get_short(user_a)}")
        print(f"Mémoire courte user_B : {self.agent.memory.get_short(user_b)}")
        print(f"Mémoire longue user_A : {self.agent.memory.get_all_long(user_a)}")
        print(f"Mémoire longue user_B : {self.agent.memory.get_all_long(user_b)}")
        # Vérification robuste : aucun message de user_B dans la mémoire de user_A, et inversement
        for m in self.agent.memory.get_short(user_a):
            assert "Utilisateur(user_B):" not in m, "Fuite mémoire user_B dans user_A !"
        for m in self.agent.memory.get_short(user_b):
            assert "Utilisateur(user_A):" not in m, "Fuite mémoire user_A dans user_B !"
        print("Isolation mémoire OK !")

    def run_all_tests(self):
        """
        Exécute tous les scénarios de test disponibles sur l'agent.
        """
        self.test_message_simple()
        self.test_message_avec_commandes()
        self.test_memoire()
        self.test_isolation_utilisateur()
