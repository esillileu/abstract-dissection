#include "w2v/w2v.h"

#include <assert.h>

int main(void)
{
    TrainingConfig config;

    training_config_defaults(&config);
    assert(training_config_validate(&config) == STATUS_OK);
    assert(status_string(STATUS_OK) != 0);
    return 0;
}
