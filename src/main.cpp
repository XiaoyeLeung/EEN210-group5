#include <Wire.h>
#include <MPU6050.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <Arduino.h>

const char *ssid = "Edith (2)";
const char *password = "tionollor";
const char *webSocketServer = "172.20.10.13";
const int webSocketPort = 8000;

MPU6050 mpu;
WebSocketsClient client;
bool wifiConnected = false;

// ======= Välj frekvenser här =======
const uint8_t MPU_RATE_DIV = 19;                 // 50 Hz (1000/(1+19)) SENSOR MPU 
const unsigned long SAMPLE_PERIOD_US = 20000;    // 50 Hz (20ms) MICROCONTROLLER
const unsigned long SEND_PERIOD_MS   = 100;      // 10 Hz (100ms) WIFI

// Batch size = hur många samples som hinner samplas på SEND_PERIOD_MS
// 50 Hz => 1 sample per 20ms => på 100ms hinner du 5 samples
const int BATCH_SIZE = 5;

// ======= Buffer for batch =======
struct Sample {
  int16_t ax, ay, az;
  int16_t gx, gy, gz;
  uint32_t t_us; // tidsstämpel (valfritt men bra)
};

Sample batch[BATCH_SIZE];
int batchIndex = 0;

unsigned long lastSampleUs = 0;
unsigned long lastSendMs = 0;

void setup() {
  Serial.begin(9600); 
  Wire.begin();

  mpu.initialize();
  // mpu.setDLPFMode(MPU6050_DLPF_BW_42); // kan aktiveras senare
  //mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_4); // ±4g
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_500); // ±500 °/s
  mpu.setRate(MPU_RATE_DIV);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.print("Connected to WiFi network with IP Address: ");
  Serial.println(WiFi.localIP());

  client.begin(webSocketServer, webSocketPort, "/ws");
  wifiConnected = true;

  lastSampleUs = micros();
  lastSendMs = millis();
}

void loop() {
  if (!wifiConnected) return;

  client.loop();

  // ===== 1) Sampla/läs sensorn i exakt takt (50 Hz) =====
  unsigned long nowUs = micros();
  if ((unsigned long)(nowUs - lastSampleUs) >= SAMPLE_PERIOD_US) {
    lastSampleUs += SAMPLE_PERIOD_US;

    // Läs sensorn
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
    mpu.getAcceleration(&ax, &ay, &az);
    mpu.getRotation(&gx, &gy, &gz);

    // Lägg in i batch-buffer
    if (batchIndex < BATCH_SIZE) {
      batch[batchIndex] = {ax, ay, az, gx, gy, gz, nowUs};
      batchIndex++;
    } else {
      // Om full: droppa senaste eller äldsta. Här droppar vi senaste (enkelt).
      // (Alternativ: ringbuffer, men detta är minsta ändring.)
    }
  }

  // ===== 2) Skicka batch med lägre frekvens (10 Hz) =====
  unsigned long nowMs = millis();
  if ((unsigned long)(nowMs - lastSendMs) >= SEND_PERIOD_MS && batchIndex > 0) {
    lastSendMs += SEND_PERIOD_MS;

    // Bygg JSON med flera samples
    String payload = "{\"samples\":[";
    for (int i = 0; i < batchIndex; i++) {
      payload += "{"
                 "\"t_us\":" + String(batch[i].t_us) +
                 ",\"ax\":" + String(batch[i].ax/16384.0f) +
                 ",\"ay\":" + String(batch[i].ay/16384.0f) +
                 ",\"az\":" + String(batch[i].az/16384.0f) +
                 ",\"gx\":" + String(batch[i].gx/65.5f) +
                 ",\"gy\":" + String(batch[i].gy/65.5f) +
                 ",\"gz\":" + String(batch[i].gz/65.5f) +
                 "}";
      if (i < batchIndex - 1) payload += ","; 
    }
    payload += "]}";

    client.sendTXT(payload);

    // Töm batch
    batchIndex = 0;

    // Debug (valfritt)
    // Serial.println(payload);
  }
}