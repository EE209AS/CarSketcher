#include<stdio.h>
#include<stdlib.h>
#include<mraa/gpio.h>
#include<mraa/pwm.h>

int main()
{

mraa_gpio_context in1;
mraa_gpio_context in2;
mraa_gpio_context in3;
mraa_gpio_context in4;

in1=mraa_gpio_init(7);
in2=mraa_gpio_init(8);
in3=mraa_gpio_init(9);
in4=mraa_gpio_init(10);

mraa_pwm_context en1;
en1=mraa_pwm_init(6);
mraa_pwm_context en2;
en2=mraa_pwm_init(5);
mraa_gpio_dir(in1,MRAA_GPIO_OUT);
mraa_gpio_dir(in2,MRAA_GPIO_OUT);
mraa_pwm_period_ms(en1,20);         
mraa_pwm_enable(en1,1);          
mraa_gpio_dir(in3,MRAA_GPIO_OUT);
mraa_gpio_dir(in4,MRAA_GPIO_OUT);
mraa_pwm_period_ms(en2,20);         
mraa_pwm_enable(en2,1);          
int i=0;                         
mraa_pwm_write(en1,1.0f);        
mraa_pwm_write(en2,1.0f);        
mraa_gpio_write(in1,1);          
mraa_gpio_write(in2,0);          
mraa_gpio_write(in3,1);          
mraa_gpio_write(in4,0);          
usleep(50000);                 
printf("back");
mraa_gpio_write(in1,0);        
mraa_gpio_write(in2,0);        
mraa_gpio_write(in3,0);        
mraa_gpio_write(in4,0);        
mraa_pwm_enable(en1,0);
mraa_pwm_enable(en2,0);                               
return 0;                        
} 
