from source.llm_agent import LLMAgent, USE_FAKE_LLM

agent = LLMAgent()
print("USE_FAKE_LLM =", USE_FAKE_LLM)
print("Action:", agent.get_action("TEST PROMPT"))
