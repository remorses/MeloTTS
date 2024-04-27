import playSound from 'play-sound'

const player = playSound({})

import fs from 'fs'
const res = await fetch('http://0.0.0.0:8888/predictions', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        input: {
            text: 'This t t s model is so good i cannot believe it, do you?.',
            speed: 1,
            speaker: 'EN-BR',
            language: 'EN',
        },
        stream: true,
    }),
})

console.log(res.status)
const json = await res.json()
console.log(json)

function dataURLtoFile(dataurl) {
    if (!dataurl) {
        return
    }
    const base64Data = dataurl.split(',')[1]
    const buffer = Buffer.from(base64Data, 'base64')
    return buffer
}

for (const [i, item] of json.output.entries()) {
    const data = dataURLtoFile(item.audio)
    const p = `out_${i}.wav`
    fs.writeFileSync(p, data)
    await new Promise((resolve, rej) =>
        player.play(p, {}, (err) => (err ? rej(err) : resolve())),
    )
}
