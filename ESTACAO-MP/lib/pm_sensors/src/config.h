#define SENSOR1
#define SENSOR2

#ifdef SENSOR1
    // #define HPMA1 
    #define SDS1
    // #define PMS1
#endif

#ifdef SENSOR2
    // #define HPMA2
    // #define SDS2
    #define PMS2 
#endif