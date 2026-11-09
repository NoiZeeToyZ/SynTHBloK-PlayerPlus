// SynTHBloK-PlayerPlus: registers this plugin's modules.
#include "plugin.hpp"

Plugin* pluginInstance;

extern Model* modelSynTHBloKPlayer;
extern Model* modelSynTHBloKPlayerView;

void init(Plugin* p) {
    pluginInstance = p;
    p->addModel(modelSynTHBloKPlayer);
    p->addModel(modelSynTHBloKPlayerView);
}
