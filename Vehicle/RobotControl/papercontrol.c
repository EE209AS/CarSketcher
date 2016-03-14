#include<stdio.h>
#include<stdlib.h>
#include<mraa/gpio.h>
#include<mraa/pwm.h>
#include<math.h>

mraa_gpio_context in1;
mraa_gpio_context in2;
mraa_gpio_context in3;
mraa_gpio_context in4;
                      
                       
mraa_pwm_context en1; 
mraa_pwm_context en2; 
#define PI 3.14159265

int rotate_right90()
{
mraa_pwm_enable(en1,1);
mraa_pwm_enable(en2,1);                                 
mraa_pwm_write(en1,1.0f);        
mraa_pwm_write(en2,1.0f);        
mraa_gpio_write(in1,1);          
mraa_gpio_write(in2,0);    
mraa_gpio_write(in3,0);    
mraa_gpio_write(in4,1);          
printf("right90");               
usleep(2300000);                 
mraa_gpio_write(in1,0);          
mraa_gpio_write(in2,0);          
mraa_gpio_write(in3,0);          
mraa_gpio_write(in4,0);          
mraa_pwm_enable(en1,0);          
mraa_pwm_enable(en2,0);          
//usleep(2300000);                 
return 0;                        
}

int rotate_left90()
{
mraa_pwm_enable(en1,1);          
mraa_pwm_enable(en2,1);
mraa_pwm_write(en1,1.0f);        
mraa_pwm_write(en2,1.0f);        
mraa_gpio_write(in1,0);          
mraa_gpio_write(in2,1);    
mraa_gpio_write(in3,0);    
mraa_gpio_write(in4,1);          
printf("left90");               
usleep(2300000);                 
mraa_gpio_write(in1,0);          
mraa_gpio_write(in2,0);          
mraa_gpio_write(in3,0);          
mraa_gpio_write(in4,0);          
mraa_pwm_enable(en1,0);          
mraa_pwm_enable(en2,0);          
//usleep(2300000);                 
return 0;                        
}

int move_front (int i)     
{

while(i!=0)
{
mraa_pwm_enable(en1,1);          
mraa_pwm_enable(en2,1);
mraa_pwm_write(en1,1.0f);
mraa_pwm_write(en2,1.0f); 
mraa_gpio_write(in1,0);          

mraa_gpio_write(in2,1);          

mraa_gpio_write(in3,0);          

mraa_gpio_write(in4,1);          

usleep(50000);                 

printf("front");

mraa_gpio_write(in1,0);        

mraa_gpio_write(in2,0);        

mraa_gpio_write(in3,0);        

mraa_gpio_write(in4,0);        

mraa_pwm_enable(en1,0);

mraa_pwm_enable(en2,0);

i--;			

}

 return 0;                 

}



int rotate_right (int i)     

{

while(i!=0)

{
mraa_pwm_enable(en1,1);          
mraa_pwm_enable(en2,1);
mraa_pwm_write(en1,1.0f);        

mraa_pwm_write(en2,1.0f);        

mraa_gpio_write(in1,0);          

mraa_gpio_write(in2,1);          

mraa_gpio_write(in3,0);          

mraa_gpio_write(in4,1);          

usleep(( 50000));     ///chagne the value accordingly  for 10 degrees            

printf("front");

mraa_gpio_write(in1,0);        

mraa_gpio_write(in2,0);        

mraa_gpio_write(in3,0);        

mraa_gpio_write(in4,0);        

mraa_pwm_enable(en1,0);

mraa_pwm_enable(en2,0);
			
i--;
}

 return 0;                 

}



int rotate_left (int i)     

{

while(i!=0)

{mraa_pwm_enable(en1,1);          
mraa_pwm_enable(en2,1);

mraa_pwm_write(en1,1.0f);        

mraa_pwm_write(en2,1.0f);        

mraa_gpio_write(in1,0);          

mraa_gpio_write(in2,1);          

mraa_gpio_write(in3,0);          

mraa_gpio_write(in4,1);          

usleep(( 50000));     ///chagne the value accordingly              

printf("front");

mraa_gpio_write(in1,0);        

mraa_gpio_write(in2,0);        

mraa_gpio_write(in3,0);        

mraa_gpio_write(in4,0);        

mraa_pwm_enable(en1,0);

mraa_pwm_enable(en2,0);

i--;			

}

 return 0;                 

}

int main(int argc, char *argv[])

{
in1=mraa_gpio_init(7); 
in2=mraa_gpio_init(8); 
in3=mraa_gpio_init(9); 
in4=mraa_gpio_init(10);


en1=mraa_pwm_init(6);                                                
en2=mraa_pwm_init(5);                                                
mraa_gpio_dir(in1,MRAA_GPIO_OUT);                                    
mraa_gpio_dir(in2,MRAA_GPIO_OUT);                                    
mraa_pwm_period_ms(en1,20);                                          
mraa_pwm_enable(en1,1);                                              
mraa_gpio_dir(in3,MRAA_GPIO_OUT);                                    
mraa_gpio_dir(in4,MRAA_GPIO_OUT);           
mraa_pwm_period_ms(en2,20);                 
mraa_pwm_enable(en2,1); 

char* ptr1 = argv[1];
char* ptr2 = argv[2]; 
char* ptr3 = argv[3]; 

float param,result;
int i= atoi(ptr1);
int j=atoi(ptr2);
int k= atoi(ptr3);

if (i == -2)
{
rotate_right(2);//because 20 degrees
}
else if( i == -1)
{
rotate_right90(); //calcualte for 90 degrees
move_front(29);   // 20 cms
rotate_left90();  //calculate for 90 degrees
}

else {


param=i/j;
//convert to our metrix, in terms of distance moved by robot//

result = atan((param) * 180 / PI);

if(result < 0)

{
rotate_left((int)(result/10));
}

else
{
rotate_right((int)(result/10));
}

float dist=sqrt(((i*i)+(j*j)));
if(k == 0)
{
dist = dist/3;
}

int num = (int) dist/(0.7) ;

move_front(num);
}                               
return 0;                        
} 
