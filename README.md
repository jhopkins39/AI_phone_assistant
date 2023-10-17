# AI_phone_assistant
Using Whisper, elevenlabs, PaLM, and Twilio to create a virtual phone assistant

The current script, when run properly creates a server endpoint, so that when you call my Twilio number (from pre-approved numbers), it directs the call to my code.
My code then asks for user input. Waits for the user to respond and ask a question. Streams to audio to a file to then pass to the whisper api to be converted to text.
Next, the text is passed as a prompt to google's PaLM model to generate a text response. This is then passed to elevenlabs to generate an audio file which is then played for the user.

Future Developments:
- creating chat functionality so the assistant and the user can have a full conversation
- fixing the stream aspect of the code(there isn't much twilio documentation for it)
- add langchain to handle the chatting with LLMs

Link to video demo:
https://youtu.be/gutPbGhlwwU
