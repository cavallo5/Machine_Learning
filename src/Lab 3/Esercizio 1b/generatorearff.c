#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main()
{
	double stepfloat, i, j;
	double result;

	printf("Inserire il passo da tenere: \n");
	scanf("%lf", &stepfloat);

	for(i=-1; i<=1; i+=stepfloat){
		for(j=-1; j<=1; j+=stepfloat){
			//printf("Passo %f ",i);
			//printf("Passo %f ",j);
			result = sin( M_PI * (pow(i,2)+pow(j,2)));
			printf("%lf,",i);
			printf("%lf,",j);
			printf("%lf,",result);
			printf("\n");
		}
	}
}