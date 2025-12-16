const API_KEY = "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";

async function run() {
  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${API_KEY}`
    },
    body: JSON.stringify({
      model: "qwen/qwen3-vl-235b-a22b-thinking",
      max_tokens: 500,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "hello my name is bharat" }
          ]
        }
      ]
    })
  });

  const result = await response.json();
//   console.log(result);
  console.log(result['choices'][0]['message']['content']);
}

run();