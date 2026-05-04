topics = [
    "AI", "Python", "programmering", "internet", "datorer",
    "Sverige", "Stockholm", "EU", "fotboll", "klimat",
    "ekonomi", "demokrati", "historia", "hälsa", "träning"
]

templates = [
    ("Vad är {topic}?", "{topic} är ett ämne som kan förklaras på ett enkelt och tydligt sätt."),
    ("Kan du förklara {topic} enkelt?", "{topic} kan beskrivas som något viktigt att förstå i vardagen eller samhället."),
    ("Varför är {topic} viktigt?", "{topic} är viktigt eftersom det påverkar hur människor lever, lär sig eller fattar beslut."),
    ("Hur fungerar {topic}?", "{topic} fungerar genom olika delar som samverkar på ett visst sätt."),
]

with open("auto_qa.txt", "w", encoding="utf-8") as f:
    for topic in topics:
        for question, answer in templates:
            f.write(f"Fråga: {question.format(topic=topic)}\n")
            f.write(f"Svar: {answer.format(topic=topic)}\n")
            f.write("Slut.\n\n")

print("Klar! Skapade auto_qa.txt")