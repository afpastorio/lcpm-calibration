#include <Arduino.h>
#include <DHT.h>
#include <FS.h>
#include <SD.h>
#include <SPI.h>
#include <SparkFunDS1307RTC.h>
#include <Wire.h>
#include <hal/hal.h>
#include <lmic.h>
#include <pm_sensors.h>
#include <driver/adc.h>
#include <driver/dac.h>
#include <SdsDustSensor.h>
#include <config.h>

// TENTATIVA DE SETAR SENSORES POR DIRETIVAS
// #if defined(HPMA1)
//     HPMA115S0 *mySensor1;
// #elif defined(SDS1)
//     SdsDustSensor mySensor1(Serial2);
// #elif defined(PMS1)
//     PMS7003 *mySensor1;
// #endif

// #if defined(HPMA2)
//     HPMA115S0 *mySensor2;
// #elif defined(SDS2)
//     SdsDustSensor mySensor2(Serial);
// #elif defined(PMS2)
//     PMS7003 *mySensor2;
// #endif

// PMS7003* myPMS;
HPMA115S0* myHPMA;
// SDS* mySDS;
SdsDustSensor mySDS(Serial);
// PmResult pm;
float sds25, sds10;

#define DHT_PIN 5
#define DHT_TYPE DHT22
DHT dht(DHT_PIN, DHT_TYPE);

#define BOARD_LED_PIN 15

#define SD_SCK 4
#define SD_MISO 32
#define SD_MOSI 2
#define SD_CS 33

// Mapa de pinos
const lmic_pinmap lmic_pins = {
    .nss = 25,
    .rxtx = LMIC_UNUSED_PIN,
    .rst = 26,
    .dio = {27, 14, 13},
};

// APPEUI da TTS em LSB da 27 d5 a1 f2 ab 54 da
static const u1_t PROGMEM APPEUI[8] = {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
void os_getArtEui(u1_t* buf) { memcpy_P(buf, APPEUI, 8); }

// DEVEUI da TTS em LSB db 09 6a f6 1e b5 2d 1d
static const u1_t PROGMEM DEVEUI[8] = {0x37, 0xE9, 0x05, 0xD0, 0x7E, 0xD5, 0xB3, 0x70};
void os_getDevEui(u1_t* buf) { memcpy_P(buf, DEVEUI, 8); }

// APPKEY da TTS em MSB e5 d4 90 e4 bb b8 3b 07 8e e3 00 5f 0b 51 87 e6
static const u1_t PROGMEM APPKEY[16] = {0xBB, 0xA6, 0xE4, 0xE5, 0x11, 0x5B, 0x17, 0x60, 0xAF, 0xFB, 0x7E, 0xF2, 0x8B, 0xF3, 0xDC, 0x61};
void os_getDevKey(u1_t* buf) { memcpy_P(buf, APPKEY, 16); }

byte packet[12];
static osjob_t sendjob;
unsigned long previous = 0;
RTC_DATA_ATTR unsigned long num_of_measurements = 0;
bool value = false;
bool debug = false;
bool go_sleep = false;

SPIClass mySPI = SPIClass(HSPI);

// Saves the LMIC structure during DeepSleep
RTC_DATA_ATTR lmic_t RTC_LMIC;

void do_send(osjob_t* j);
void printValues(PM_Sensor* sensor);
void printHex2(unsigned v);
void onEvent(ev_t ev);
void printTime();
void saveData();
void SaveLMICToRTC();
void LoadLMICFromRTC();
void GoDeepSleep();

void printHex2(unsigned v) {
    v &= 0xff;
    if (v < 16)
        Serial.print('0');
    Serial.print(v, HEX);
}

void onEvent(ev_t ev) {
    Serial.print(os_getTime());
    Serial.print(": ");
    switch (ev) {
        case EV_SCAN_TIMEOUT:
            if(debug)
                Serial.println(F("EV_SCAN_TIMEOUT"));
            break;
        case EV_BEACON_FOUND:
            if(debug)
                Serial.println(F("EV_BEACON_FOUND"));
            break;
        case EV_BEACON_MISSED:
            if(debug)
                Serial.println(F("EV_BEACON_MISSED"));
            break;
        case EV_BEACON_TRACKED:
            if(debug)
                Serial.println(F("EV_BEACON_TRACKED"));
            break;
        case EV_JOINING:
            if(debug)
                Serial.println(F("EV_JOINING"));
            break;
        case EV_JOINED:
            if(debug) {
                Serial.println(F("EV_JOINED"));
                
                u4_t netid = 0;
                devaddr_t devaddr = 0;
                u1_t nwkKey[16];
                u1_t artKey[16];
                LMIC_getSessionKeys(&netid, &devaddr, nwkKey, artKey);
                Serial.print("netid: ");
                Serial.println(netid, DEC);
                Serial.print("devaddr: ");
                Serial.println(devaddr, HEX);
                Serial.print("AppSKey: ");
                for (size_t i = 0; i < sizeof(artKey); ++i) {
                    if (i != 0)
                        Serial.print("-");
                    printHex2(artKey[i]);
                }
                Serial.println("");
                Serial.print("NwkSKey: ");
                for (size_t i = 0; i < sizeof(nwkKey); ++i) {
                    if (i != 0)
                        Serial.print("-");
                    printHex2(nwkKey[i]);
                }
                Serial.println();
            }
            LMIC_setLinkCheckMode(0);
            break;
        case EV_JOIN_FAILED:
            if(debug)
                Serial.println(F("EV_JOIN_FAILED"));
            break;
        case EV_REJOIN_FAILED:
            if(debug)
                Serial.println(F("EV_REJOIN_FAILED"));
            break;
        case EV_TXCOMPLETE:
            if(debug) {
                Serial.println(F("EV_TXCOMPLETE (includes waiting for RX windows)"));
                if (LMIC.txrxFlags & TXRX_ACK)
                    Serial.println(F("Received ack"));
                if (LMIC.dataLen) {
                    Serial.print(F("Received "));
                    Serial.print(LMIC.dataLen);
                    Serial.println(F(" bytes of payload"));
                }
            }
            go_sleep = true;
            break;
        case EV_LOST_TSYNC:
            if(debug)
                Serial.println(F("EV_LOST_TSYNC"));
            break;
        case EV_RESET:
            if(debug)
                Serial.println(F("EV_RESET"));
            break;
        case EV_RXCOMPLETE:
            if(debug)
                Serial.println(F("EV_RXCOMPLETE"));
            break;
        case EV_LINK_DEAD:
            if(debug)
                Serial.println(F("EV_LINK_DEAD"));
            break;
        case EV_LINK_ALIVE:
            if(debug)
                Serial.println(F("EV_LINK_ALIVE"));
            break;
        case EV_TXSTART:
            if(debug)
                Serial.println(F("EV_TXSTART"));
            break;
        case EV_TXCANCELED:
            if(debug)
                Serial.println(F("EV_TXCANCELED"));
            break;
        case EV_RXSTART:
            break;
        case EV_JOIN_TXCOMPLETE:
            if(debug)
                Serial.println(F("EV_JOIN_TXCOMPLETE: no JoinAccept"));
            break;
        default:
            if(debug)
                Serial.print(F("Unknown event: "));
                Serial.println((unsigned)ev);
            break;
    }
}

void do_send(osjob_t* j) {
    // Check if there is not a current TX/RX job running
    if (LMIC.opmode & OP_TXRXPEND) {
        if(debug)
            Serial.println(F("OP_TXRXPEND, not sending"));
    } else {
        uint16_t PM25_1, PM10_1, PM25_2, PM10_2;

        PM25_1 = int(myHPMA->getPM25()*100.0);
        PM10_1 = int(myHPMA->getPM10()*100.0);
        // PM25_2 = int(myPMS->getPM25()*100.0);
        // PM10_2 = int(myPMS->getPM10()*100.0);
        // PM25_2 = int(mySDS->getPM25()*100.0);
        // PM10_2 = int(mySDS->getPM10()*100.0);
        PM25_2 = int(sds25*100.0);
        PM10_2 = int(sds10*100.0);
        float h = dht.readHumidity();
        float t = dht.readTemperature();
        uint16_t H_int = int(h * 100);
        uint16_t T_int = int(t * 100);
        packet[0] = highByte(PM25_1);
        packet[1] = lowByte(PM25_1);
        packet[2] = highByte(PM10_1);
        packet[3] = lowByte(PM10_1);
        packet[4] = highByte(PM25_2);
        packet[5] = lowByte(PM25_2);
        packet[6] = highByte(PM10_2);
        packet[7] = lowByte(PM10_2);
        packet[8] = highByte(T_int);
        packet[9] = lowByte(T_int);
        packet[10] = highByte(H_int);
        packet[11] = lowByte(H_int);

        LMIC_setTxData2(1, packet, sizeof(packet), 0);
    }
}

void setup() {
    adc_power_off();
    dac_i2s_disable();
    gpio_deep_sleep_hold_dis();

    // mySDS = new SDS(&Serial, PASSIVE, 9600);
    mySDS.begin();
    mySDS.setContinuousWorkingPeriod();
    mySDS.setQueryReportingMode();
    mySDS.sleep();

    // myPMS = new PMS7003(&Serial, PASSIVE, 9600);
    myHPMA = new HPMA115S0(&Serial2, PASSIVE, 9600);
    pinMode(BOARD_LED_PIN, OUTPUT);
    delay(1000);

    rtc.begin();  // Call rtc.begin() to initialize the library
    dht.begin();

    pinMode(SD_MOSI, INPUT_PULLUP);
    mySPI.begin(SD_SCK, SD_MISO, SD_MOSI);
    if (!SD.begin(SD_CS, mySPI)) {
        if(debug)
            Serial.println("Card Mount Failed");
        return;
    } else {
        if(debug)
            Serial.println("Ok");
    }

    uint8_t cardType = SD.cardType();

    if (cardType == CARD_NONE) {
        if(debug)
            Serial.println("No SD card attached");
        return;
    }
    if(debug) {
        Serial.print("SD Card Type: ");
        if (cardType == CARD_MMC) {
            Serial.println("MMC");
        } else if (cardType == CARD_SD) {
            Serial.println("SDSC");
        } else if (cardType == CARD_SDHC) {
            Serial.println("SDHC");
        } else {
            Serial.println("UNKNOWN");
        }
        uint64_t cardSize = SD.cardSize() / (1024 * 1024);
        Serial.printf("SD Card Size: %lluMB\n", cardSize);
        Serial.printf("Used space: %lluMB\n", SD.usedBytes() / (1024 * 1024));

        Serial.println("Starting...");
    }

    // Configuração de canais para a rede ATC
    // LMIC_enableChannel(0);
    // LMIC_enableChannel(1);
    // LMIC_enableChannel(2);
    // LMIC_enableChannel(3);
    // LMIC_enableChannel(4);
    // LMIC_enableChannel(5);
    // LMIC_enableChannel(6);
    // LMIC_enableChannel(7);
    // LMIC init
    os_init();
    // Reset the MAC state. Session and pending data transfers will be discarded.
    LMIC_reset();

    LMIC_setAdrMode(0);
    LMIC_selectSubBand(1);
    LMIC.dn2Dr = DR_SF9;
    LMIC_setClockError(MAX_CLOCK_ERROR * 1 / 100);

    if (RTC_LMIC.seqnoUp != 0) {
        LoadLMICFromRTC();
        LMIC_setLinkCheckMode(0);
    }

    if(debug)
        Serial.println("Started!");
}

void loop() {
    os_runloop_once();
    rtc.update();
    if (millis() - previous > 30000) {
        digitalWrite(BOARD_LED_PIN, value);
        // myPMS->Read();
        myHPMA->Read();

        // mySDS->Read();
        mySDS.wakeup();
        delay(5000);
        PmResult pm = mySDS.queryPm();
        sds25 = pm.pm25;
        sds10 = pm.pm10;
        mySDS.sleep();

        saveData();
        if (value)
            value = false;
        else
            value = true;
        do_send(&sendjob);
        previous = millis();
    }
    if(go_sleep) {
        go_sleep = false;
        SaveLMICToRTC();
        LMIC_shutdown();
        GoDeepSleep();
    }
}

void printTime() {
    Serial.print(String(rtc.hour()) + ":");  // Print hour
    if (rtc.minute() < 10)
        Serial.print('0');                     // Print leading '0' for minute
    Serial.print(String(rtc.minute()) + ":");  // Print minute
    if (rtc.second() < 10)
        Serial.print('0');               // Print leading '0' for second
    Serial.print(String(rtc.second()));  // Print second

    if (rtc.is12Hour())  // If we're in 12-hour mode
    {
        // Use rtc.pm() to read the AM/PM state of the hour
        if (rtc.pm())
            Serial.print(" PM");  // Returns true if PM
        else
            Serial.print(" AM");
    }

    Serial.print(" | ");

    // Few options for printing the day, pick one:
    Serial.print(rtc.dayStr());  // Print day string
    // Serial.print(rtc.dayC()); // Print day character
    // Serial.print(rtc.day()); // Print day integer (1-7, Sun-Sat)
    Serial.print(" - ");
#ifdef PRINT_USA_DATE
    Serial.print(String(rtc.month()) + "/" +  // Print month
                 String(rtc.date()) + "/");   // Print date
#else
    Serial.print(String(rtc.date()) + "/" +   // (or) print date
                 String(rtc.month()) + "/");  // Print month
#endif
    Serial.println(String(rtc.year()));  // Print year
}

void printValues(PM_Sensor* sensor) {
    int PM25, PM10;
    // PM1 = sensor->getPM1();
    PM25 = sensor->getPM25();
    PM10 = sensor->getPM10();

    // Serial.print("PM1.0: ");
    // Serial.print(PM1);
    Serial.print(" PM2.5: ");
    Serial.print(PM25);
    Serial.print(" PM10: ");
    Serial.println(PM10);
}

void saveData() {
    num_of_measurements++;
    String toSave = String(rtc.date()) + "/" + String(rtc.month()) + "/" + String(rtc.year()) + ";" +
                    String(rtc.hour()) + ":" + String(rtc.minute()) + ":" + String(rtc.second()) + ";" +
                    String(num_of_measurements) + ";";

    float h = dht.readHumidity();
    float t = dht.readTemperature();
    float PM25_1, PM10_1, PM25_2, PM10_2;
    PM25_1 = myHPMA->getPM25();
    PM10_1 = myHPMA->getPM10();
    // PM25_2 = myPMS->getPM25();
    // PM10_2 = myPMS->getPM10();

    // PM25_2 = mySDS->getPM25();
    // PM10_2 = mySDS->getPM10();
    PM25_2 = sds25;
    PM10_2 = sds10;

    if (debug) {
        Serial.print(F(" -- Humidity: "));
        Serial.print(h);
        Serial.print(F("%  Temperature: "));
        Serial.print(t);
        Serial.println(F("°C "));
    }

    toSave += String(t, 4) + ";" + String(h, 4) + ";";
    toSave += String(PM25_1, 1)+ ";" + String(PM10_1, 1) + ";";
    toSave += String(PM25_2, 1)+ ";" + String(PM10_2, 1) + ";";

    String filename = "/" + String(rtc.date()) + String(rtc.month()) + String(rtc.year()) + ".csv";
    File myFile = SD.open(filename, "a+");
    myFile.println(toSave);

    myFile.close();
}

void SaveLMICToRTC() {
    RTC_LMIC = LMIC;
}

void LoadLMICFromRTC() {
    LMIC_setSession(RTC_LMIC.netid, RTC_LMIC.devaddr, RTC_LMIC.nwkKey, RTC_LMIC.artKey);
    LMIC.seqnoUp = RTC_LMIC.seqnoUp;
    LMIC.seqnoDn = RTC_LMIC.seqnoDn;
}

void GoDeepSleep() {
    gpio_deep_sleep_hold_en();
    esp_sleep_enable_timer_wakeup(600 * 1000000);
    esp_deep_sleep_start();
}