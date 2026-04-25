from grid07.content_engine import generate_post
from grid07.personas import BOT_A, BOT_B, BOT_C
import json

# runs generate_post for each bot and prints the result as JSON
for bot in [BOT_A, BOT_B, BOT_C]:
    result = generate_post(bot)
    print(json.dumps({
        "bot_id"      : result.bot_id,
        "topic"       : result.topic,
        "post_content": result.post_content,
    }, indent=2))
    print()
