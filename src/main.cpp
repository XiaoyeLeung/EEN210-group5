#include <Wire.h>
#include <MPU6050.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <Adafruit_BusIO_Register.h>

// Public internet, allow port in firewall
// Replace with your network credentials
const char *ssid = "Edith (2)"; //
const char *password = "tionollor";

// Replace with your WebSocket server address
const char *webSocketServer = "your WiFi IP Address";

const int webSocketPort = 8000;
const char *webSocketPath = "/";
MPU6050 mpu; // Define the sensor
WebSocketsClient client;
// SocketIOClient socketIO;

bool wifiConnected = false;

void setup()
{
  Serial.begin(9600);
  Wire.begin();
  mpu.initialize();

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("Connected to WiFi network with IP Address: ");
  Serial.println(WiFi.localIP());
  client.begin(webSocketServer, webSocketPort, "/ws");
  Serial.println(client.isConnected());
  wifiConnected = true;
}

void loop()
{
  if (wifiConnected)
  {
    client.loop();

    // Get sensor data
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
    mpu.getAcceleration(&ax, &ay, &az);
    mpu.getRotation(&gx, &gy, &gz);

    // Convert data to a JSON string
    String payload = "{\"acceleration_x\":" + String(ax) +
                     ",\"acceleration_y\":" + String(ay) +
                     ",\"acceleration_z\":" + String(az) +
                     ",\"gyroscope_x\":" + String(gx) +
                     ",\"gyroscope_y\":" + String(gy) +
                     ",\"gyroscope_z\":" + String(gz) + "}";

    Serial.println("Skiikar....");
    // server address, port and URL
    // Send data via WebSocket
    client.sendTXT(payload);
    client.loop();

    delay(10); // Adjust delay as needed
  }
}
