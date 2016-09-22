/*
 * environment.h
 *
 *  Created on: Jan 28, 2016
 *      Author: kalin
 */

#ifndef sbpl_adim_multirep_adaptive_environment_h
#define sbpl_adim_multirep_adaptive_environment_h

#include <sbpl_adaptive/headers.h>

#include <sbpl_adaptive/macros.h>

namespace adim {

struct EnvStateData
{
    EnvStateData()
    {
        goaldata = NULL;
        HashTableSize = 32 * 1024;
        goalHashEntry = NULL;
        startHashEntry = NULL;
    }

    AbstractGoal_t *goaldata;

    // start and goal entries
    AdaptiveHashEntry *goalHashEntry;
    AdaptiveHashEntry *startHashEntry;

    // hash tables
    size_t HashTableSize;
    std::vector<std::vector<std::vector<AdaptiveHashEntry *>>> HashTables;

    // vector that maps from stateID to coords
    std::vector<AdaptiveHashEntry *> StateID2HashEntry;
};

SBPL_CLASS_FORWARD(MultiRepAdaptiveDiscreteSpaceInformation)

class MultiRepAdaptiveDiscreteSpaceInformation :
    public AdaptiveDiscreteSpaceInformation
{
public:

    MultiRepAdaptiveDiscreteSpaceInformation();

    virtual ~MultiRepAdaptiveDiscreteSpaceInformation();

    /// inserts a new hash entry into the hash table and updates its stateID
    /// accordingly returns the stateID
    size_t InsertHashEntry(AdaptiveHashEntry *entry, size_t binID);

    AdaptiveHashEntry *GetState(size_t stateID);

    const EnvStateData *getEnvStateDataPtr() { return &data_; }

    const void *getEnvDataPtr() { return env_data_.get(); }

    bool RegisterFullDRepresentation(
        AdaptiveStateRepresentationPtr fullD_representation);

    bool RegisterRepresentation(AdaptiveStateRepresentationPtr rep);

    int GetFullDRepresentationID() { return fulld_representation_->getID(); }

    AdaptiveStateRepresentation *GetFullDRepresentation();
    const AdaptiveStateRepresentation *GetFullDRepresentation() const;

    AdaptiveStateRepresentation *GetRepresentation(int dimID);
    const AdaptiveStateRepresentation *GetRepresentation(int dimID) const;

    // convenient wrappers that (unsafely) downwcast to a specified type for the
    // representation
    template <typename T> T *GetFullDRepresentation();
    template <typename T> const T *GetFullDRepresentation() const;
    template <typename T> T *GetRepresentation(int dimID);
    template <typename T> const T *GetRepresentation(int dimID) const;

    int SetStartCoords(int dimID, const void *representation_specific_data);
    int SetStartConfig(int dimID, const void *representation_specific_data);
    int SetGoalCoords(int dimID, const void *goal_representation_specific_discrete_data);
    int SetGoalConfig(int dimID, const void *goal_representation_specific_continuous_data);

    int SetAbstractGoal(AbstractGoal_t *abstract_goal);

    void updateBestTracked(int StateID, int costToGoal);

    virtual bool ProjectToFullD(
        const void *local_state_data,
        int fromID,
        std::vector<int> &proj_stateIDs,
        int adPathIdx = 0);

    virtual bool Project(
        const void *hd_state_data,
        int fromID,
        int toID,
        std::vector<int> &proj_stateIDs,
        int adPathIdx = 0);

    /// \name Required Public Functions From AdaptiveDiscreteSpaceInformation
    ///@{

    bool isExecutablePath(const std::vector<int> &stateIDV);

    virtual void GetSuccs_Track(
        int SourceStateID,
        std::vector<int> *SuccIDV,
        std::vector<int> *CostV);

    virtual void GetSuccs_Track(
        int SourceStateID,
        int expansion_step,
        std::vector<int> *SuccIDV,
        std::vector<int> *CostV);

    virtual void GetSuccs_Plan(
        int SourceStateID,
        std::vector<int> *SuccIDV,
        std::vector<int> *CostV);

    virtual void GetSuccs_Plan(
        int SourceStateID,
        int expansion_step,
        std::vector<int> *SuccIDV,
        std::vector<int> *CostV);

    virtual void GetPreds_Track(
        int TargetStateID,
        std::vector<int> *PredIDV,
        std::vector<int> *CostV);

    virtual void GetPreds_Track(
        int TargetStateID,
        int expansion_step,
        std::vector<int> *PredIDV,
        std::vector<int> *CostV);

    virtual void GetPreds_Plan(
        int TargetStateID,
        std::vector<int> *PredIDV,
        std::vector<int> *CostV);

    virtual void GetPreds_Plan(
        int TargetStateID,
        int expansion_step,
        std::vector<int> *PredIDV,
        std::vector<int> *CostV);

    ///@}


protected:

    int BestTracked_StateID;
    int BestTracked_Cost;

    void InsertMetaGoalHashEntry(AdaptiveHashEntry *entry);

    AdaptiveStateRepresentationPtr fulld_representation_;
    std::vector<AdaptiveStateRepresentationPtr> representations_;

    EnvStateData data_;

    std::shared_ptr<void> env_data_;
};

inline
AdaptiveStateRepresentation *
MultiRepAdaptiveDiscreteSpaceInformation::GetFullDRepresentation()
{
    return fulld_representation_.get();
}

inline
const AdaptiveStateRepresentation *
MultiRepAdaptiveDiscreteSpaceInformation::GetFullDRepresentation() const
{
    return fulld_representation_.get();
}

inline
AdaptiveStateRepresentation *
MultiRepAdaptiveDiscreteSpaceInformation::GetRepresentation(int dimID)
{
    if (dimID < 0 || dimID >= representations_.size()) {
        return NULL;
    }
    return representations_[dimID].get();
}

inline
const AdaptiveStateRepresentation *
MultiRepAdaptiveDiscreteSpaceInformation::GetRepresentation(int dimID) const
{
    if (dimID < 0 || dimID >= representations_.size()) {
        return NULL;
    }
    return representations_[dimID].get();
}

template <typename T> T *
MultiRepAdaptiveDiscreteSpaceInformation::GetFullDRepresentation()
{
    return (T *)fulld_representation_.get();
}

template <typename T> const T *
MultiRepAdaptiveDiscreteSpaceInformation::GetFullDRepresentation() const
{
    return (const T *)fulld_representation_.get();
}

template <typename T> T *
MultiRepAdaptiveDiscreteSpaceInformation::GetRepresentation(int dimID)
{
    if (dimID < 0 || dimID >= representations_.size()) {
        return nullptr;
    }
    return (T *)representations_[dimID].get();
}

template <typename T> const T *
MultiRepAdaptiveDiscreteSpaceInformation::GetRepresentation(int dimID) const
{
    if (dimID < 0 || dimID >= representations_.size()) {
        return nullptr;
    }
    return (const T *)representations_[dimID].get();
}

inline
void MultiRepAdaptiveDiscreteSpaceInformation::updateBestTracked(
    int StateID,
    int costToGoal)
{
    if (costToGoal < BestTracked_Cost) {
        BestTracked_Cost = costToGoal;
        BestTracked_StateID = StateID;
        SBPL_INFO("BestSeenState: %d @ %d", BestTracked_StateID, BestTracked_Cost);
    }
}

} // namespace adim

#endif
