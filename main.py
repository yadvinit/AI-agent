from crewai import Agent, Task, Crew
from gemini_llm import GeminiLLM

# Create Gemini instance
llm = GeminiLLM(api_key="AIzaSyAs-YSD2jDVw3sopRYyDlWl3PNbqhnW6eE")

# Define agents
researcher = Agent(
    role='Researcher',
    goal='Find motivational quotes',
    backstory='Expert researcher.',
    llm=llm
)

writer = Agent(
    role='image Creator',
    goal='create a poster',
    backstory='Professional poster creator.',
    llm=llm
)

# Define tasks
task_research = Task(
    description='Find motivation quotes',
    agent=researcher,
    expected_output='motivational things'
)

task_write = Task(
    description='Write it on poster ',
    agent=writer,
    expected_output='A poster of quote.'
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task_research, task_write]
)

result = crew.kickoff()
print(result)
