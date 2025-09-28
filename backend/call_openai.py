from openai import OpenAI

client = OpenAI(
  api_key="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"
)

response = client.responses.create(
    model="gpt-5-nano",
    input="The input will be a jumbled string without spaces, like 'thisiprety'. "
          "Turn it into a short, natural prose sentence. "
          "Example: 'thisiprety' → 'This is pretty.'\n\n"
          "Now process this: thisiprety",
)

print(response.output_text)
