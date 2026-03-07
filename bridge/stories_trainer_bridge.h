#ifndef STORIES_TRAINER_BRIDGE_H
#define STORIES_TRAINER_BRIDGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ANEStoriesTrainerHandle ANEStoriesTrainerHandle;

typedef struct ANEStoriesStepStats {
    uint32_t step;
    float loss;
    double step_ms;
    uint32_t compiles;
    uint32_t restart_required;
} ANEStoriesStepStats;

ANEStoriesTrainerHandle *ane_bridge_stories_open(const char *model_path,
                                                 const char *model_key,
                                                 const char *data_path,
                                                 size_t input_bytes,
                                                 size_t output_bytes,
                                                 uint32_t total_steps,
                                                 float lr,
                                                 bool ane_extras,
                                                 uint32_t compile_budget);

int ane_bridge_stories_step(ANEStoriesTrainerHandle *h, ANEStoriesStepStats *stats);
int ane_bridge_stories_save_checkpoint(ANEStoriesTrainerHandle *h, const char *path);
int ane_bridge_stories_load_checkpoint(ANEStoriesTrainerHandle *h, const char *path);
void ane_bridge_stories_close(ANEStoriesTrainerHandle *h);
int ane_bridge_stories_last_error(char *buf, size_t n);

#ifdef __cplusplus
}
#endif

#endif
