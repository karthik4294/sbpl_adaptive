/*
 * Copyright (c) 2011, Kalin Gochev
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/** \author Kalin Gochev */

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sbpl_adaptive_collision_checking/sbpl_collision_model.h>

namespace sbpl_adaptive_collision_checking {

struct URDFModelCoords_t : ModelCoords_t
{
    Eigen::Affine3d root;
    std::map<std::string, std::vector<double>> coordmap;
    std::vector<std::string> collision_links;
    std::vector<std::string> contact_links;

    bool getCoords(
        const std::string &name,
        std::vector<double> &p_coords) const;

    bool getCoord(const std::string &name, double &p_coord, int idx = 0) const;

    double getCoord(const std::string &name, int idx = 0) const;

    void set(const std::string &name, const std::vector<double> &p_coords);

    static void print(const URDFModelCoords_t &c);

    static void print(const Eigen::Affine3d &t);

    static void updateWith(
        URDFModelCoords_t &target,
        const URDFModelCoords_t &source);

    static double getMaxJointDistance(
        const URDFModelCoords_t &from,
        const URDFModelCoords_t &to);
};

struct SegmentPair
{
    std::string root;
    std::string tip;
    KDL::Segment segment;

    SegmentPair(
        const KDL::Segment &p_segment,
        const std::string &p_root,
        const std::string &p_tip)
    :
        segment(p_segment), root(p_root), tip(p_tip)
    {
    }
};

struct AttachedObject_t
{
    std::string name;
    std::vector<Sphere> spheres;
};

class URDFCollisionModel : public SBPLCollisionModel
{
public:

    URDFCollisionModel();
    ~URDFCollisionModel();

    URDFModelCoords_t getRandomCoordinates() const;

    URDFModelCoords_t getDefaultCoordinates() const;

    virtual bool initFromURDF(
        const std::string &urdf_string,
        const std::string &srdf_string);

    virtual bool initFromParam(const std::string &robot_desc_param_name);

    virtual bool computeSpheresFromURDFModel(
        double res,
        const std::vector<std::string> &ignore_collision_links,
        const std::vector<std::string> &contact_links);

    virtual void autoIgnoreSelfCollisions();
    virtual void autoIgnoreSelfCollisions(const URDFModelCoords_t &coords);

    void printIgnoreSelfCollisionLinkPairs();

    bool InitializeChains(const std::vector<std::string> &chain_tip_link_names);

    bool checkLimits(const URDFModelCoords_t &coords) const;

    bool checkSelfCollisions(const URDFModelCoords_t &coords) const;

    std::vector<std::pair<std::string, std::string>> getSelfCollisions(
        const URDFModelCoords_t &coords) const;

    bool getModelCollisionSpheres(
        const URDFModelCoords_t &coords,
        std::vector<Sphere> &spheres) const;

    bool getModelContactSpheres(
        const URDFModelCoords_t &coords,
        std::vector<Sphere> &spheres) const;

    bool getModelPathCollisionSpheres(
        const URDFModelCoords_t &coords0,
        const URDFModelCoords_t &coords1,
        int steps,
        std::vector<Sphere> &spheres) const;

    bool getModelPathContactSpheres(
        const URDFModelCoords_t &coords0,
        const URDFModelCoords_t &coords1,
        int steps,
        std::vector<Sphere> &spheres) const;

    virtual bool getInterpolatedCoordinates(
        const URDFModelCoords_t &coords0,
        const URDFModelCoords_t &coords1,
        double t,
        URDFModelCoords_t &interp) const;
    virtual bool getInterpolatedPath(
        const URDFModelCoords_t &coords0,
        const URDFModelCoords_t &coords1,
        double resolution,
        std::vector<URDFModelCoords_t> &path) const;
    virtual bool getInterpolatedPath(
        const URDFModelCoords_t &coords0,
        const URDFModelCoords_t &coords1,
        int steps,
        std::vector<URDFModelCoords_t> &path) const;

    bool initFromFile(std::string fname);

    void addContactSpheres(const std::string &link_name, const std::vector<Sphere> &s);

    void addCollisionSpheres(const std::string &link_name, const std::vector<Sphere> &s);

    void addContactSphere(const std::string &link_name, Sphere s);

    void addCollisionSphere(const std::string &link_name, Sphere s);

    // get more advanced mesh visualization when available
    visualization_msgs::MarkerArray getModelSelfCollisionVisualization(
        const URDFModelCoords_t &coords,
        const std::string &frame_id,
        const std::string &ns,
        const std_msgs::ColorRGBA &col,
        int &idx) const;

    visualization_msgs::MarkerArray getModelBasicVisualizationByLink(
        const URDFModelCoords_t &coords,
        std::string frame_id,
        std::string ns,
        int &idx) const;
    visualization_msgs::MarkerArray getModelBasicVisualization(
        const URDFModelCoords_t &coords,
        std::string frame_id,
        std::string ns,
        std_msgs::ColorRGBA col,
        int &idx) const;
    visualization_msgs::MarkerArray getModelVisualization(
        const URDFModelCoords_t &coords,
        std::string frame_id,
        std::string ns,
        std_msgs::ColorRGBA col,
        int &idx) const;
    visualization_msgs::MarkerArray getAttachedObjectsVisualization(
        const URDFModelCoords_t &coords,
        std::string frame_id,
        std::string ns,
        std_msgs::ColorRGBA col,
        int &idx) const;

    bool computeGroupIK(
        const std::string &group_name,
        const Eigen::Affine3d &ee_pose_map,
        const URDFModelCoords_t &seed,
        URDFModelCoords_t &sol,
        bool bAllowApproxSolutions = false,
        int n_attempts = 0,
        double time_s = 0);

    bool computeCOM(
        const URDFModelCoords_t &coords,
        KDL::Vector &COM,
        double &mass) const;

    bool computeChainTipPoses(
        const URDFModelCoords_t &coords,
        std::map<std::string, Eigen::Affine3d> &tip_poses);

    bool getLinkGlobalTransform(
        const URDFModelCoords_t &coords,
        const std::string &link_name,
        Eigen::Affine3d &tfm) const;

    bool attachObjectToLink(
        const std::string &link_name,
        const Eigen::Affine3d &pose,
        const shapes::Shape &object,
        const std::string &object_name,
        double res);

    void PrintModelInfo() const;

    void addIgnoreSelfCollisionLinkPair(
        const std::pair<std::string, std::string> pair);
    void addIgnoreSelfCollisionLinkPairs(
        const std::vector<std::pair<std::string, std::string>> pairs);

    /// \name Reimplemented Public Functions
    ///@{
    bool checkLimits(const ModelCoords_t &coord) const override;

    bool getModelCollisionSpheres(
        const ModelCoords_t &coords,
        std::vector<Sphere> &spheres) const override;

    bool getModelContactSpheres(
        const ModelCoords_t &coords,
        std::vector<Sphere> &spheres) const override;

    bool getModelContactSpheres(
        const ModelCoords_t &coords,
        const std::string &link_name,
        std::vector<Sphere> &spheres) const override;

    bool getModelPathCollisionSpheres(
        const ModelCoords_t &coords0,
        const ModelCoords_t &coords1,
        int steps,
        std::vector<Sphere> &spheres) const override;

    bool getModelPathContactSpheres(
        const ModelCoords_t &coords0,
        const ModelCoords_t &coords1,
        int steps,
        std::vector<Sphere> &spheres) const override;

    visualization_msgs::MarkerArray getModelVisualization(
        const ModelCoords_t &coords,
        const std::string &frame_id,
        const std::string &ns,
        const std_msgs::ColorRGBA &col,
        int &idx) const override;
    ///@}

protected:

    ros::AsyncSpinner* spinner;

    boost::shared_ptr<urdf::Model> urdf_;

    int num_active_joints_;
    int num_coords_;

    bool bFixedRoot;

    std::vector<std::string> links_with_collision_spheres_;
    std::vector<std::string> links_with_contact_spheres_;

    std::vector<std::pair<std::string, std::string>> self_collision_ignore_pairs_;

    std::map<std::string, std::vector<Sphere>> collision_spheres_;
    std::map<std::string, std::vector<Sphere>> contact_spheres_;

    std::map<std::string, std::vector<AttachedObject_t>> attached_objects_;

    robot_model_loader::RobotModelLoaderPtr rm_loader_;
    robot_model::RobotModelPtr robot_model_;
    robot_state::RobotStatePtr robot_state_;

    KDL::Tree kdl_tree_;
    std::map<std::string, KDL::Chain> kdl_chains_;
    std::map<std::string, SegmentPair> segments_;
    std::map<std::string, std::unique_ptr<KDL::ChainIkSolverPos_LMA>> kdl_ik_solvers_;
    std::map<std::string, std::unique_ptr<KDL::ChainFkSolverPos_recursive>> kdl_fk_solvers_;

    bool loadKDLModel();

    void addChildren(const KDL::SegmentMap::const_iterator segment);

    bool updateFK(const URDFModelCoords_t &coords) const;

    bool hasIgnoreSelfPair(std::string link1, std::string link2) const;

    bool getLinkCollisionSpheres(
        const URDFModelCoords_t &coords,
        std::string link_name,
        std::vector<Sphere> &spheres) const;
    bool getLinkCollisionSpheres_CurrentState(
        std::string link_name,
        std::vector<Sphere> &spheres) const;

    bool getLinkContactSpheres(
        const URDFModelCoords_t &coords,
        std::string link_name,
        std::vector<Sphere> &spheres) const;
    bool getLinkContactSpheres_CurrentState(
        std::string link_name,
        std::vector<Sphere> &spheres) const;

    bool getLinkAttachedObjectsSpheres(
        const std::string &link_name,
        const Eigen::Affine3d link_tfm,
        std::vector<Sphere> &spheres) const;

    bool computeCOMRecurs(
        const KDL::SegmentMap::const_iterator &current_seg,
        const URDFModelCoords_t &coords,
        const KDL::Frame &tf,
        double &m,
        KDL::Vector &com) const;
    bool computeChainTipPosesRecurs(
        const KDL::SegmentMap::const_iterator &current_seg,
        const URDFModelCoords_t &joint_positions,
        const KDL::Frame &tf,
        std::map<std::string, KDL::Frame> &tip_frames);

    bool initRobotModelFromURDF(
        const std::string &urdf_string,
        const std::string &srdf_string);

    bool setJointVariables(
        std::string joint_name,
        const std::vector<double> &variables);

    bool hasAttachedObject(
        const std::string &link_name,
        const std::string &object_name) const;
    bool hasAttachedObjects(const std::string &link_name) const;
    const std::vector<AttachedObject_t> getAttachedObjects(
            const std::string &link_name) const;

    bool attachObject(
        const std::string &link_name,
        const AttachedObject_t &obj);
};

//////////////////////////////////////
// URDFModelCoords_t Implementation //
//////////////////////////////////////

inline
bool URDFModelCoords_t::getCoords(
    const std::string &name,
    std::vector<double> &p_coords) const
{
    auto jnt = coordmap.find(name);
    if (jnt == coordmap.end()) {
        return false;
    }
    p_coords = jnt->second;
    return true;
}

inline
bool URDFModelCoords_t::getCoord(
    const std::string &name,
    double &p_coord,
    int idx) const
{
    auto jnt = coordmap.find(name);
    if (jnt == coordmap.end()) {
        return false;
    }
    if (idx >= jnt->second.size()) {
        return false;
    }
    p_coord = jnt->second[idx];
    return true;
}

inline
double URDFModelCoords_t::getCoord(const std::string &name, int idx) const
{
    auto jnt = coordmap.find(name);
    if (jnt == coordmap.end()) {
        ROS_ERROR("name %s not found in coordmap", name.c_str());
        throw SBPL_Exception();
        return 0;
    }
    if (idx >= jnt->second.size()) {
        ROS_ERROR("index %d out of bounds [0,%zu)", idx, jnt->second.size());
        throw SBPL_Exception();
        return false;
    }
    return jnt->second[idx];
}

inline
void URDFModelCoords_t::set(
    const std::string &name,
    const std::vector<double> &p_coords)
{
    coordmap[name] = p_coords;
}

inline
void URDFModelCoords_t::print(const URDFModelCoords_t &c)
{
    printf("root: [%.3f %.3f %.3f %.3f]\n", c.root(0, 0), c.root(0, 1), c.root(0, 2), c.root(0, 3));
    printf("root: [%.3f %.3f %.3f %.3f]\n", c.root(1, 0), c.root(1, 1), c.root(1, 2), c.root(1, 3));
    printf("root: [%.3f %.3f %.3f %.3f]\n", c.root(2, 0), c.root(2, 1), c.root(2, 2), c.root(2, 3));
    printf("root: [%.3f %.3f %.3f %.3f]\n", c.root(3, 0), c.root(3, 1), c.root(3, 2), c.root(3, 3));
    for (auto it = c.coordmap.begin(); it != c.coordmap.end();it++) {
        printf("%s : ", it->first.c_str());
        for (double v : it->second) {
            printf("%.3f ", v);
        }
        printf("\n");
    }
}

inline
void URDFModelCoords_t::print(const Eigen::Affine3d &t)
{
    printf("[%.3f %.3f %.3f %.3f]\n", t(0, 0), t(0, 1), t(0, 2), t(0, 3));
    printf("[%.3f %.3f %.3f %.3f]\n", t(1, 0), t(1, 1), t(1, 2), t(1, 3));
    printf("[%.3f %.3f %.3f %.3f]\n", t(2, 0), t(2, 1), t(2, 2), t(2, 3));
    printf("[%.3f %.3f %.3f %.3f]\n", t(3, 0), t(3, 1), t(3, 2), t(3, 3));
}

inline
void URDFModelCoords_t::updateWith(
    URDFModelCoords_t &target,
    const URDFModelCoords_t &source)
{
    for (auto jnt = source.coordmap.begin(); jnt != source.coordmap.end();
            jnt++)
    {
        target.set(jnt->first, jnt->second);
    }
}

inline
double URDFModelCoords_t::getMaxJointDistance(
    const URDFModelCoords_t &from,
    const URDFModelCoords_t &to)
{
    double max_dist = 0.0;
    for (auto it = to.coordmap.begin(); it != to.coordmap.end(); it++) {
        std::vector<double> seed_val;
        if (!from.getCoords(it->first, seed_val)) {
            ROS_ERROR("%s not found in [from]!", it->first.c_str());
            throw SBPL_Exception();
        }
        for (int v = 0; v < seed_val.size(); v++) {
            double diff = fabs(angles::shortest_angular_distance(
                    seed_val[v], it->second[v]));
            if (diff > max_dist) {
                //ROS_WARN("MaxDiff: %.3f deg @ %s", angles::to_degrees(diff), it->first.c_str());
                max_dist = diff;
            }
        }
    }
    return max_dist;
}

///////////////////////////////////////
// URDFCollisionModel Implementation //
///////////////////////////////////////

inline
void URDFCollisionModel::addContactSpheres(
    const std::string &link_name,
    const std::vector<Sphere> &s)
{
    for (auto sp : s) {
        addContactSphere(link_name, sp);
    }
}

inline
void URDFCollisionModel::addCollisionSpheres(
    const std::string &link_name,
    const std::vector<Sphere> &s)
{
    for (auto sp : s) {
        addCollisionSphere(link_name, sp);
    }
}

inline
void URDFCollisionModel::addContactSphere(
    const std::string &link_name,
    Sphere s)
{
    std::map<std::string, std::vector<Sphere>>::iterator it = contact_spheres_.find(link_name);
    if (it == contact_spheres_.end()) {
        s.link_name_ = link_name;
        s.name_ = link_name + "_0";
        contact_spheres_[link_name] = {s};
        links_with_contact_spheres_.push_back(link_name);
        ROS_INFO("Added contact sphere 1 for link %s", link_name.c_str());
    }
    else {
        s.link_name_ = link_name;
        s.name_ = link_name + "_" + std::to_string(it->second.size());
        it->second.push_back(s);
        ROS_INFO("Added contact sphere %d for link %s", (int )it->second.size(), link_name.c_str());
    }
}

inline
void URDFCollisionModel::addCollisionSphere(const std::string &link_name, Sphere s)
{
    std::map<std::string, std::vector<Sphere>>::iterator it = collision_spheres_.find(link_name);
    if (it == collision_spheres_.end()) {
        s.link_name_ = link_name;
        s.name_ = link_name + "_0";
        collision_spheres_[link_name] = {s};
        links_with_collision_spheres_.push_back(link_name);
    }
    else {
        s.link_name_ = link_name;
        s.name_ = link_name + "_" + std::to_string(it->second.size());
        it->second.push_back(s);
    }
}

inline
void URDFCollisionModel::addIgnoreSelfCollisionLinkPair(
    const std::pair<std::string, std::string> pair)
{
    if (!hasIgnoreSelfPair(pair.first, pair.second)) {
        self_collision_ignore_pairs_.push_back(pair);
    }
}

inline
void URDFCollisionModel::addIgnoreSelfCollisionLinkPairs(
    const std::vector<std::pair<std::string, std::string>> pairs)
{
    for (auto pair : pairs) {
        addIgnoreSelfCollisionLinkPair(pair);
    }
}

inline
bool URDFCollisionModel::getModelCollisionSpheres(
    const ModelCoords_t &coords,
    std::vector<Sphere> &spheres) const
{
    const URDFModelCoords_t &c = dynamic_cast<const URDFModelCoords_t&>(coords);
    return getModelCollisionSpheres(c, spheres);
}

inline
bool URDFCollisionModel::getModelContactSpheres(
    const ModelCoords_t &coords,
    std::vector<Sphere> &spheres) const
{
    const URDFModelCoords_t &c = dynamic_cast<const URDFModelCoords_t&>(coords);
    return getModelContactSpheres(c, spheres);
}

inline
bool URDFCollisionModel::getModelContactSpheres(
    const ModelCoords_t &coords,
    const std::string &link_name,
    std::vector<Sphere> &spheres) const
{
    const URDFModelCoords_t &c = dynamic_cast<const URDFModelCoords_t&>(coords);
    return getLinkContactSpheres(c, link_name, spheres);
}

inline
bool URDFCollisionModel::getModelPathCollisionSpheres(
    const ModelCoords_t &coords0,
    const ModelCoords_t &coords1,
    int steps,
    std::vector<Sphere> &spheres) const
{
    const URDFModelCoords_t &c0 = dynamic_cast<const URDFModelCoords_t&>(coords0);
    const URDFModelCoords_t &c1 = dynamic_cast<const URDFModelCoords_t&>(coords1);
    return getModelPathCollisionSpheres(c0, c1, steps, spheres);
}

inline
bool URDFCollisionModel::getModelPathContactSpheres(
    const ModelCoords_t &coords0,
    const ModelCoords_t &coords1,
    int steps,
    std::vector<Sphere> &spheres) const
{
    const URDFModelCoords_t &c0 = dynamic_cast<const URDFModelCoords_t&>(coords0);
    const URDFModelCoords_t &c1 = dynamic_cast<const URDFModelCoords_t&>(coords1);
    return getModelPathContactSpheres(c0, c1, steps, spheres);
}

inline
bool URDFCollisionModel::checkLimits(const ModelCoords_t &coord) const
{
    const URDFModelCoords_t &c = dynamic_cast<const URDFModelCoords_t&>(coord);
    return checkLimits(c);
}

inline
visualization_msgs::MarkerArray URDFCollisionModel::getModelVisualization(
    const ModelCoords_t &coords,
    const std::string &frame_id,
    const std::string &ns,
    const std_msgs::ColorRGBA &col,
    int &idx) const
{
    const URDFModelCoords_t &c = dynamic_cast<const URDFModelCoords_t&>(coords);
    return getModelVisualization(c, frame_id, ns, col, idx);
}

inline
bool URDFCollisionModel::hasIgnoreSelfPair(
    std::string link1,
    std::string link2) const
{
    for (int k = 0; k < self_collision_ignore_pairs_.size(); k++) {
        std::pair<std::string, std::string> p = self_collision_ignore_pairs_[k];
        if (link1 == p.first && link2 == p.second) {
            return true;
        }
        else if (link1 == p.second && link2 == p.first) {
            return true;
        }
    }
    return false;
}

} // namespace sbpl_adaptive_collision_checking