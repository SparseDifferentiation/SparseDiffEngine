
#ifndef INT_DOUBLE_PAIR_H
#define INT_DOUBLE_PAIR_H

typedef struct int_double_pair
{
    int col;
    double val;
} int_double_pair;

int_double_pair *new_int_double_pair_array(int size);
void set_int_double_pair_array(int_double_pair *pair, int *ints, double *doubles,
                               int size);
void free_int_double_pair_array(int_double_pair *array);
void sort_int_double_pair_array(int_double_pair *array, int size);

#endif /* INT_DOUBLE_PAIR_H */
