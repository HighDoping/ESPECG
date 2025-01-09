// esp32 v2.0.17
#include <SPI.h>
#include <TFT_eSPI.h> // Hardware-specific library v2.5.43

#include <SD.h>
#include <FS.h>
////////////////////TFT//////////////////////
#define ILI9341_DRIVER

#define TFT_MISO 12 // (leave TFT SDO disconnected if other SPI devices share MISO)
#define TFT_MOSI 13
#define TFT_SCLK 14
#define TFT_CS 15 // Chip select control pin
#define TFT_DC 2  // Data Command control pin
#define TFT_RST 4 // Reset pin (could connect to RST pin)

// Optional touch screen chip select
#define TOUCH_CS 5 // Chip select pin (T_CS) of touch screen

#define LOAD_GLCD  // Font 1. Original Adafruit 8 pixel font needs ~1820 bytes in FLASH
#define LOAD_FONT2 // Font 2. Small 16 pixel high font, needs ~3534 bytes in FLASH, 96 characters
#define LOAD_FONT4 // Font 4. Medium 26 pixel high font, needs ~5848 bytes in FLASH, 96 characters
#define LOAD_FONT6 // Font 6. Large 48 pixel font, needs ~2666 bytes in FLASH, only characters 1234567890:-.apm
#define LOAD_FONT7 // Font 7. 7 segment 48 pixel font, needs ~2438 bytes in FLASH, only characters 1234567890:.
#define LOAD_FONT8 // Font 8. Large 75 pixel font needs ~3256 bytes in FLASH, only characters 1234567890:-.
#define LOAD_GFXFF // FreeFonts. Include access to the 48 Adafruit_GFX free fonts FF1 to FF48 and custom fonts

#define SMOOTH_FONT

// TFT SPI clock frequency
// #define SPI_FREQUENCY  20000000
// #define SPI_FREQUENCY  27000000
#define SPI_FREQUENCY 40000000
// #define SPI_FREQUENCY  80000000

// Optional reduced SPI frequency for reading TFT
#define SPI_READ_FREQUENCY 16000000

// SPI clock frequency for touch controller
#define SPI_TOUCH_FREQUENCY 2500000


// Define custom pin
const int LO_N = 35;
const int LO_P = 34;

// Define SD card parameters
const int SD_CS = 17;
const int SD_FREQ = 4000000;
File dataFile;

// display variables
int x_count = 0;
int map_data_old = 0;

// logging variables
uint64_t sample_count = 0;
bool lead_off = false;

// Define ADC parameters
const uint8_t ADC_PIN = 33;
uint8_t adc_pins[] = {ADC_PIN};              // Analog input pin
const uint8_t ADC_WIDTH = 12;                 // ADC resolution
const adc_attenuation_t ADC_ATTEN = ADC_11db; // ADC attenuation

// Define timer interrupt parameters
const int SAMPLE_RATE = 20000; // Sample rate in Hz (e.g., 1000 samples per second)
#define CONVERSIONS_PER_PIN 5
// Define interrupt buffer variables

#define BUFFER_SIZE 256

uint8_t adc_pins_count = sizeof(adc_pins) / sizeof(uint8_t);
volatile bool adc_coversion_done = false;
adc_continuous_data_t *adc_result_local = NULL;

volatile uint32_t buffer[BUFFER_SIZE];
volatile bool newDataAvailable = false;
volatile uint8_t bufferIndex = 0;

class CircularBuffer
{
public:
  CircularBuffer(int size) : size_(size), buffer_(new int[size]), head_(0), tail_(0), count_(0) {}

  ~CircularBuffer()
  {
    delete[] buffer_;
  }

  void push(int data)
  {
    buffer_[head_] = data;
    head_ = (head_ + 1) % size_;
    if (count_ < size_)
    {
      count_++;
    }
    else
    {
      tail_ = (tail_ + 1) % size_;
    }
  }

  int *read()
  {
    int *data = new int[count_];
    for (int i = 0; i < count_; i++)
    {
      data[i] = buffer_[(tail_ + i) % size_];
    }
    return data;
  }

private:
  int size_;
  int *buffer_;
  int head_;
  int tail_;
  int count_;
};

// ISR Function that will be triggered when ADC conversion is done
void ARDUINO_ISR_ATTR adcComplete()
{
  adc_coversion_done = true;
  buffer[bufferIndex] = adc_result_local[0].avg_read_mvolts;
  bufferIndex=bufferIndex+1;
  newDataAvailable = true;
  adc_coversion_done = false;
}

TFT_eSPI tft = TFT_eSPI(); // Invoke custom library

// Function prototypes
bool lead_off_detect();
void tft_update_line(int data);
void sd_write(int data);
bool openSDFile();
bool closeSDFile();

void setup()
{
  Serial.begin(115200);
  // if (openSDFile())
  // {
  //   // Write the CSV header
  //   dataFile.println("sample_count,data");
  // }
  Serial.println("Init TFT");
  tft.init();
  tft.setRotation(3);
  tft.setTextSize(1);
  tft.fillScreen(TFT_BLACK);
  tft.setCursor(0, 0);
  tft.setTextColor(TFT_YELLOW, TFT_BLACK);
  tft.println("ESP32 ECG");

  pinMode(LO_P, INPUT);
  pinMode(LO_N, INPUT);
  pinMode(ADC_PIN,INPUT);

  pinMode(0, INPUT);

  // Configure ADC
  Serial.println("Init ADC");
  analogContinuousSetWidth(ADC_WIDTH);
  analogContinuousSetAtten(ADC_ATTEN);
  analogContinuous(adc_pins, adc_pins_count, CONVERSIONS_PER_PIN, SAMPLE_RATE, &adcComplete);
  if (analogContinuousStart())
  {
    Serial.println("Start");
  }
  else
  {
    Serial.println("ADC Init Failed");
  }

  // // check if model loaded fine
  // while (!tf.begin(ecgmodel).isOk())
  //   Serial.println(tf.exception.toString());
  
}

void loop()
{
  lead_off_detect();
  // Process the data if new data is available
  if (newDataAvailable)
  {
    // Copy data from buffer for processing
    uint32_t localBuffer[BUFFER_SIZE];
    // stop ISR
    noInterrupts();
    memcpy(localBuffer, (const void *)buffer, sizeof(buffer));
    interrupts();
    newDataAvailable = false;
    int bufferIndex_old = bufferIndex;
    bufferIndex = 0;
    // Process the data in localBuffer
    for (int i = 0; i < bufferIndex_old; i++)
    {
      sample_count++;
      tft_update_line(localBuffer[i]);
    }
    Serial.println("New data");

  }

  // Close the file if the button is pressed
  if (digitalRead(0) == LOW & dataFile)
  {
    Serial.println("Close File");
    closeSDFile();
  }
}

bool lead_off_detect()
{
  if (digitalRead(LO_P) == HIGH)
  {
    // Serial.println("LA off");
    lead_off = true;
  }
  else if (digitalRead(LO_N) == HIGH)
  {
    // Serial.println("RA off");
    lead_off = true;
  }
  else
  {
    lead_off = false;
  }
  return lead_off;
}

void tft_update_line(int data)
{
  if (x_count > 320)
  {
    x_count = 0;
  }
  // tft.setCursor(0, 8);
  // tft.setTextColor(TFT_YELLOW, TFT_RED);
  // tft.fillRect(0, 8, 320, 8, TFT_BLACK);
  int shift = 16;
  int map_data = map(data, 0, 4095, 200 + shift, 0 + shift);
  tft.fillRect(x_count, shift, 10, 201, TFT_BLACK);
  tft.drawLine(x_count - 1, map_data_old, x_count, map_data, TFT_GREEN);
  x_count++;
  map_data_old = map_data;
}

void sd_write(int data)
{
  // Open the CSV file in append mode
  // Check if the file was opened successfully
  if (dataFile)
  {
    // Write the data to the file with the timestamp
    dataFile.print(sample_count);
    dataFile.print(",");
    dataFile.println(data);
    // Close the file
    // Serial.println("Data written to data.csv");
  }
  else
  {
    // Serial.println("Error opening data.csv for writing!");
  }
}
bool openSDFile()
{
  if (!SD.begin(SD_CS, SPI, SD_FREQ))
  {
    Serial.println("SD card initialization failed!");
  }
  // Open the CSV file in append mode
  dataFile = SD.open("/data.csv", FILE_WRITE);
  // Check if the file was opened successfully
  if (!dataFile)
  {
    Serial.println("Error opening data.csv");
    return false;
  }
  return true;
}

bool closeSDFile()
{
  if (dataFile)
  {
    dataFile.close();
    return true;
  }
  SD.end();
  return false;
}

