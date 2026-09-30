#!/usr/bin/env bash
# ==============================================================================
# Script de automação para compilar e gravar o firmware no Arduino Nano 33 BLE
# ==============================================================================
set -e

SKETCH_DIR="firmware/MosquitoPresence"
FQBN="arduino:mbed_nano:nano33ble"

echo "=== Gravador Automático de Firmware - Mosquito Wingbeat ==="

# 1. Verificar se arduino-cli está disponível
if ! command -v arduino-cli &> /dev/null; then
    echo "[!] arduino-cli não encontrado no PATH do sistema."
    echo "Deseja baixar e instalar o arduino-cli localmente nesta pasta? (s/n)"
    read -r resposta
    if [[ "$resposta" =~ ^[sS]$ ]]; then
        echo "[*] Baixando arduino-cli..."
        curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | BINDIR=. sh
        ARDUINO_CLI="./arduino-cli"
    else
        echo "Instalação cancelada. Instale o arduino-cli ou use a Arduino IDE gráfica."
        exit 1
    fi
else
    ARDUINO_CLI="arduino-cli"
fi

# 2. Atualizar índice e instalar core da placa
echo "[*] Verificando núcleo de placas Mbed Nano ($FQBN)..."
$ARDUINO_CLI core update-index
if ! $ARDUINO_CLI core list | grep -q "arduino:mbed_nano"; then
    echo "[*] Instalando núcleo arduino:mbed_nano..."
    $ARDUINO_CLI core install arduino:mbed_nano
fi

# 3. Detectar porta serial automaticamente ou permitir informar manualmente
PORT="$1"
if [ -z "$PORT" ]; then
    echo "[*] Procurando placa Arduino conectada..."
    DETECTED_PORT=$($ARDUINO_CLI board list | grep -i "nano.*33.*ble" | awk '{print $1}' | head -n 1)
    if [ -n "$DETECTED_PORT" ]; then
        PORT="$DETECTED_PORT"
        echo "[✓] Placa detectada automaticamente na porta: $PORT"
    else
        echo "[!] Nenhuma placa identificada automaticamente."
        echo "Portas seriais disponíveis no sistema:"
        ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
        echo ""
        read -p "Digite a porta da sua placa (ex: /dev/ttyACM0): " PORT
    fi
fi

if [ -z "$PORT" ]; then
    echo "Erro: Nenhuma porta serial informada. Conecte a placa e tente novamente."
    exit 1
fi

# 4. Compilar sketch
echo "[*] Compilando sketch em $SKETCH_DIR para $FQBN..."
$ARDUINO_CLI compile --fqbn "$FQBN" "$SKETCH_DIR"

# 5. Fazer upload para a placa
echo "[*] Gravando firmware na porta $PORT..."
$ARDUINO_CLI upload --fqbn "$FQBN" --port "$PORT" "$SKETCH_DIR"

echo "=== SUCESSO! Firmware gravado no Arduino com sucesso! ==="
echo "Para monitorar as previsões da placa, use:"
echo "  $ARDUINO_CLI monitor -p $PORT -c baudrate=115200"
