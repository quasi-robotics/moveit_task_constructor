/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016, Kentaro Wada.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Author: Michael 'v4hn' Goerner */

#include "execute_task_solution_capability.h"

#include <moveit/moveit_cpp/moveit_cpp.h>
#include <moveit/plan_execution/plan_execution.h>
#include <moveit/trajectory_processing/trajectory_tools.h>
#include <moveit/kinematic_constraints/utils.h>
#include <moveit/move_group/capability_names.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/utils/message_checks.h>
#include <boost/algorithm/string/join.hpp>

#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>

namespace {

// TODO: move to moveit::core::RobotModel
const moveit::core::JointModelGroup* findJointModelGroup(const moveit::core::RobotModel& model,
                                                         const std::vector<std::string>& joints) {
	std::set<std::string> joint_set(joints.begin(), joints.end());

	const std::vector<const moveit::core::JointModelGroup*>& jmgs = model.getJointModelGroups();

	for (const moveit::core::JointModelGroup* jmg : jmgs) {
		const std::vector<std::string>& jmg_joints = jmg->getJointModelNames();
		std::set<std::string> jmg_joint_set(jmg_joints.begin(), jmg_joints.end());

		// return group if sets agree on all active joints
		if (std::includes(jmg_joint_set.begin(), jmg_joint_set.end(), joint_set.begin(), joint_set.end())) {
			std::set<std::string> difference;
			std::set_difference(jmg_joint_set.begin(), jmg_joint_set.end(), joint_set.begin(), joint_set.end(),
			                    std::inserter(difference, difference.begin()));
			unsigned int acceptable = 0;
			for (const std::string& diff_joint : difference) {
				const moveit::core::JointModel* diff_jm = model.getJointModel(diff_joint);
				if (diff_jm->isPassive() || diff_jm->getMimic() || diff_jm->getType() == moveit::core::JointModel::FIXED)
					++acceptable;
			}
			if (difference.size() == acceptable)
				return jmg;
		}
	}

	return nullptr;
}
}  // namespace

static const rclcpp::Logger LOGGER = rclcpp::get_logger("moveit_task_constructor_visualization.execute_task_solution");

namespace move_group {

ExecuteTaskSolutionCapability::ExecuteTaskSolutionCapability() : MoveGroupCapability("ExecuteTaskSolution") {}

void ExecuteTaskSolutionCapability::initialize() {
	// configure the action server
	as_ = rclcpp_action::create_server<moveit_task_constructor_msgs::action::ExecuteTaskSolution>(
	    context_->moveit_cpp_->getNode(), "execute_task_solution",
	    ActionServerType::GoalCallback(std::bind(&ExecuteTaskSolutionCapability::handleNewGoal, this,
	                                             std::placeholders::_1, std::placeholders::_2)),
	    ActionServerType::CancelCallback(
	        std::bind(&ExecuteTaskSolutionCapability::preemptCallback, this, std::placeholders::_1)),
	    ActionServerType::AcceptedCallback(
	        std::bind(&ExecuteTaskSolutionCapability::goalCallback, this, std::placeholders::_1)));
}

void ExecuteTaskSolutionCapability::goalCallback(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<ExecuteTaskSolutionAction>> goal_handle) {
	auto result = std::make_shared<moveit_task_constructor_msgs::action::ExecuteTaskSolution::Result>();

	const auto& goal = goal_handle->get_goal();
	if (!context_->plan_execution_) {
		result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::CONTROL_FAILED;
		goal_handle->abort(result);
		return;
	}

	plan_execution::ExecutableMotionPlan plan;
	if (!constructMotionPlan(goal->solution, plan))
		result->error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_MOTION_PLAN;
	else {
		RCLCPP_INFO(LOGGER, "Executing TaskSolution");
		result->error_code = context_->plan_execution_->executeAndMonitor(plan);
	}

	if (result->error_code.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
		goal_handle->succeed(result);
	else if (result->error_code.val == moveit_msgs::msg::MoveItErrorCodes::PREEMPTED)
		goal_handle->canceled(result);
	else
		goal_handle->abort(result);
}

rclcpp_action::CancelResponse ExecuteTaskSolutionCapability::preemptCallback(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<ExecuteTaskSolutionAction>> goal_handle) {
	if (context_->plan_execution_)
		context_->plan_execution_->stop();
	return rclcpp_action::CancelResponse::ACCEPT;
}

bool ExecuteTaskSolutionCapability::constructMotionPlan(const moveit_task_constructor_msgs::msg::Solution& solution,
                                                        plan_execution::ExecutableMotionPlan& plan) {
	moveit::core::RobotModelConstPtr model = context_->planning_scene_monitor_->getRobotModel();

	moveit::core::RobotState state(model);
	{
		planning_scene_monitor::LockedPlanningSceneRO scene(context_->planning_scene_monitor_);
		state = scene->getCurrentState();
	}

	// Using TOTG to parameterize the trajectory segments
	moveit_msgs::msg::RobotTrajectory merged_trajectory_msg;

	if (solution.sub_trajectory.empty()) {
		RCLCPP_ERROR_STREAM(LOGGER, "Passed solution have empty vector of sub-trajectory messages");
		return false;
	} else {
		merged_trajectory_msg = solution.sub_trajectory.at(0).trajectory;
		for (size_t i = 1; i < solution.sub_trajectory.size(); i++) {
			const auto& trajectory_points = solution.sub_trajectory.at(i).trajectory.joint_trajectory.points;
			merged_trajectory_msg.joint_trajectory.points.insert(merged_trajectory_msg.joint_trajectory.points.end(),
			                                                     trajectory_points.begin(), trajectory_points.end());
		}
	}

	// Make sure the timestamps are increasing to prevent having negative time when converting it to RobotTrajectory
	// TOTG will change it later
	auto time_from_start = rclcpp::Duration::from_seconds(1);
	for (auto& point : merged_trajectory_msg.joint_trajectory.points) {
		point.time_from_start = time_from_start;
		time_from_start = time_from_start + rclcpp::Duration::from_seconds(1);
	}

	plan.plan_components_.emplace_back();
	plan_execution::ExecutableTrajectory& exec_traj = plan.plan_components_.back();

	// define individual variable for use in closure below
	std::string description = "combined solutions of:\n";
	for (const auto& sub_trajectory : solution.sub_trajectory)
		description += "subsolution " + std::to_string(sub_trajectory.info.id) + " of stage " +
		               std::to_string(sub_trajectory.info.stage_id) + "\n";

	exec_traj.description_ = description;

	const moveit::core::JointModelGroup* group = nullptr;
	{
		std::vector<std::string> joint_names(merged_trajectory_msg.joint_trajectory.joint_names);
		joint_names.insert(joint_names.end(), merged_trajectory_msg.multi_dof_joint_trajectory.joint_names.begin(),
		                   merged_trajectory_msg.multi_dof_joint_trajectory.joint_names.end());
		if (!joint_names.empty()) {
			group = findJointModelGroup(*model, joint_names);
			if (!group) {
				RCLCPP_ERROR_STREAM(LOGGER, "Could not find JointModelGroup that actuates {"
				                                << boost::algorithm::join(joint_names, ", ") << "}");
				return false;
			}
			RCLCPP_DEBUG(LOGGER, "Using JointModelGroup '%s' for execution", group->getName().c_str());
		}
	}
	exec_traj.trajectory_ = std::make_shared<robot_trajectory::RobotTrajectory>(model, group);
	exec_traj.trajectory_->setRobotTrajectoryMsg(state, merged_trajectory_msg);
	trajectory_processing::TimeOptimalTrajectoryGeneration totg_trajectory_timing_algorithm;
	if (!totg_trajectory_timing_algorithm.computeTimeStamps(*exec_traj.trajectory_)) {
		RCLCPP_ERROR(LOGGER, "Calling computeTimeStamps on the trajectory failed");
		return false;
	}

	/* TODO add action feedback and markers */
	exec_traj.effect_on_success_ = [this, sub_trajectories = solution.sub_trajectory,
	                                description](const plan_execution::ExecutableMotionPlan*) {
		for (const auto& sub_trajectory : sub_trajectories) {
			if (!moveit::core::isEmpty(sub_trajectory.scene_diff)) {
				RCLCPP_DEBUG_STREAM(LOGGER, "apply effect of " << description);
				if (!context_->moveit_cpp_->getPlanningSceneMonitor()->newPlanningSceneMessage(sub_trajectory.scene_diff))
					return false;
			}
		}
		return true;
	};

	for (const auto& sub_trajectory : solution.sub_trajectory) {
		if (!moveit::core::isEmpty(sub_trajectory.scene_diff.robot_state) &&
		    !moveit::core::robotStateMsgToRobotState(sub_trajectory.scene_diff.robot_state, state, true)) {
			RCLCPP_ERROR(LOGGER, "invalid intermediate robot state in scene diff of SubTrajectory");
			return false;
		}
	}

	return true;
}

}  // namespace move_group

#include <class_loader/class_loader.hpp>
CLASS_LOADER_REGISTER_CLASS(move_group::ExecuteTaskSolutionCapability, move_group::MoveGroupCapability)
