/*
 * representation.h
 *
 *  Created on: Feb 8, 2016
 *      Author: kalin
 */

#ifndef _ADAPTIVE_REPRESENTATION3D_H_
#define _ADAPTIVE_REPRESENTATION3D_H_

#include <sbpl_adaptive/discrete_space_information/multirep_adaptive_3d/multirep_adaptive_3d.h>
#include <sbpl_adaptive/macros.h>
#include <sbpl_adaptive/representation.h>

namespace adim {

SBPL_CLASS_FORWARD(AdaptiveStateRepresentation3D)

class AdaptiveStateRepresentation3D : public AdaptiveStateRepresentation
{
public:

    AdaptiveStateRepresentation3D(
        MultiRepAdaptiveDiscreteSpaceInformation3DPtr env,
        bool executable,
        std::string description);

    virtual ~AdaptiveStateRepresentation3D() { }

    virtual std::vector<AdaptiveSphere3D> getSpheresForState(int StateID) = 0;

    virtual std::vector<AdaptiveSphere3D> getUpgradeSpheresForState(
        int StateID) = 0;

    virtual std::vector<AdaptiveSphere3D> getTunnelSpheresForState(
        int StateID,
        int costToGoal) = 0;

    virtual std::vector<Position3D> getSpherePositionsForState(int StateID) = 0;

protected:

    MultiRepAdaptiveDiscreteSpaceInformation3DPtr env3d_;
};

inline
AdaptiveStateRepresentation3D::AdaptiveStateRepresentation3D(
    MultiRepAdaptiveDiscreteSpaceInformation3DPtr env,
    bool executable,
    std::string description)
:
    AdaptiveStateRepresentation(env, executable, description),
    env3d_(env)
{

}

} // namespace adim

#endif
