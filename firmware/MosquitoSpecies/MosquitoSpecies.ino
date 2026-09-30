/*
  Experimental 20-species acoustic classifier: Nano 33 BLE Sense / Sense Rev2.
  Board: Arduino Mbed OS Nano Boards -> Arduino Nano 33 BLE.
  Uses the built-in PDM microphone. No extra ML/DSP library is needed.
  Keep ALL .h files next to this sketch. Serial Monitor: 115200 baud.
  Scores are model outputs, not a guarantee that a detected species is correct.
  Details, measured accuracy and limits: README.md in this folder.
*/
#include <PDM.h>
#include "StreamingFeatures.h"
#include "PresenceModel.h"
#include "SpeciesModel.h"
#include "SpeciesDecision.h"

mosquito::StreamingFeatures frontend;
mosquito::SpeciesConfirmation confirmation;
float featureVector[mosquito::kFeatureCount];
float speciesScores[mosquito::kSpeciesCount];
// Defaults come from calibration sources, without using the test sources.
// Lowering either threshold emits more labels and can increase mistakes.
const float presenceThreshold=mosquito::kModerateThreshold;
const float speciesThreshold=mosquito::kSpeciesConfidenceThreshold;

const uint16_t ringSize=4096;
volatile int16_t pcmRing[ringSize];
volatile uint16_t ringHead=0,ringTail=0;
volatile uint32_t lostSamples=0;
uint32_t observedLostSamples=0,clippedSamples=0,windowSamples=0;
uint16_t largestSample=0;

void onPDMdata() {
  int16_t chunk[256];
  while(PDM.available()>0) {
    int count=PDM.available();
    if(count>static_cast<int>(sizeof(chunk))) count=sizeof(chunk);
    const int bytesRead=PDM.read(chunk,count);
    if(bytesRead<=0) break;
    count=bytesRead/sizeof(int16_t);
    for(int i=0;i<count;++i) {
      const uint16_t next=(ringHead+1)&(ringSize-1);
      if(next==ringTail) { ++lostSamples;continue; }
      pcmRing[ringHead]=chunk[i];ringHead=next;
    }
  }
}

void setup() {
  pinMode(LED_BUILTIN,OUTPUT);
  digitalWrite(LED_BUILTIN,LOW);
  Serial.begin(115200);
  const uint32_t start=millis();
  while(!Serial && millis()-start<3000) {}
  PDM.onReceive(onPDMdata);
  PDM.setBufferSize(512);
  PDM.setGain(20);
  if(!PDM.begin(1,mosquito::kSampleRate)) {
    Serial.println("ERRO: microfone PDM indisponivel.");
    while(true) { digitalWrite(LED_BUILTIN,!digitalRead(LED_BUILTIN));delay(500); }
  }
  Serial.println("# Modelo experimental: 20 especies; 16 kHz; janela 0,992 s; concordancia 2/3.");
  Serial.print("# limiar_presenca=");Serial.print(presenceThreshold,6);
  Serial.print(" limiar_classe=");Serial.println(speciesThreshold,6);
  Serial.println("tempo_ms,estado,candidata,identificada,escore_classe,limiar_classe,escore_presenca,audio_ok,amostras_perdidas,inferencia_us");
}

void loop() {
  int16_t sample=0;
  bool available=false,discontinuity=false;
  noInterrupts();
  if(lostSamples!=observedLostSamples) {
    observedLostSamples=lostSamples;
    ringTail=ringHead;discontinuity=true;
  }
  if(ringTail!=ringHead) {
    sample=pcmRing[ringTail];ringTail=(ringTail+1)&(ringSize-1);available=true;
  }
  interrupts();
  if(discontinuity) {
    frontend.reset();confirmation.reset();
    windowSamples=clippedSamples=largestSample=0;
    digitalWrite(LED_BUILTIN,LOW);
    Serial.print("# AUDIO_INTERROMPIDO: amostras perdidas=");Serial.println(observedLostSamples);
  }
  if(!available) return;
  const uint16_t magnitude=sample<0?-static_cast<int32_t>(sample):sample;
  if(magnitude>largestSample) largestSample=magnitude;
  if(magnitude>=32700) ++clippedSamples;
  ++windowSamples;
  if(!frontend.push(sample,featureVector)) return;

  // Capture safeguards only; rates on a physical microphone remain unmeasured.
  const bool qualityOK=largestSample>=32 && clippedSamples*100<=windowSamples;
  const uint32_t inferenceStart=micros();
  const float presenceScore=mosquito::presenceScore(featureVector);
  mosquito::speciesProbabilities(featureVector,speciesScores);
  const int8_t best=mosquito::bestSpecies(speciesScores);
  const float classScore=speciesScores[best];
  const bool presenceCandidate=qualityOK && presenceScore>=presenceThreshold;
  const int8_t eligible=presenceCandidate && classScore>=speciesThreshold?best:-1;
  int8_t identified=-1;
  if(qualityOK) identified=confirmation.update(eligible);
  else confirmation.reset();
  const uint32_t inferenceUs=micros()-inferenceStart;

  const char* state=!qualityOK?"AUDIO_INVALIDO":!presenceCandidate?"SEM_EVIDENCIA":
                    identified<0?"INCERTO":"IDENTIFICACAO_PROVISORIA";
  digitalWrite(LED_BUILTIN,identified>=0?HIGH:LOW);
  Serial.print(millis());Serial.print(',');Serial.print(state);Serial.print(',');
  Serial.print(presenceCandidate?mosquito::kSpeciesNames[best]:"-");Serial.print(',');
  Serial.print(identified>=0?mosquito::kSpeciesNames[identified]:"-");Serial.print(',');
  Serial.print(classScore,6);Serial.print(',');Serial.print(speciesThreshold,6);Serial.print(',');
  Serial.print(presenceScore,6);Serial.print(',');Serial.print(qualityOK?1:0);Serial.print(',');
  Serial.print(observedLostSamples);Serial.print(',');Serial.println(inferenceUs);
  windowSamples=clippedSamples=largestSample=0;
}
