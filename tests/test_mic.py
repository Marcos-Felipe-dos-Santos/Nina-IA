import pyaudio

p = pyaudio.PyAudio()
print("Dispositivos de entrada disponíveis:")
for i in range(p.get_device_count()):
    d = p.get_device_info_by_index(i)
    if d["maxInputChannels"] > 0:
        nome = d["name"]
        canais = d["maxInputChannels"]
        print(f"  [{i}] {nome} — canais: {canais}")
p.terminate()