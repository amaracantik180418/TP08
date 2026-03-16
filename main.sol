// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title TensorProxima_08
/// @notice On-chain registry for AI training run metadata: epochs, loss snapshots, and checkpoint hashes.
/// @dev Used by training pipelines to anchor run identifiers and metrics without storing full datasets.
///      Proxima anchor suite; indexers expect configHash = keccak256(abi.encode(hyperparams)).

contract TensorProxima_08 {

    // -------------------------------------------------------------------------
    // EVENTS
    // -------------------------------------------------------------------------

    event RunRegistered(
        bytes32 indexed runId,
        address indexed submitter,
        uint16 epochCount,
        bytes32 configHash,
        uint256 anchoredAt
    );
    event EpochRecorded(
        bytes32 indexed runId,
        uint32 indexed epochIndex,
        uint256 lossScaled,
        bytes32 gradientRoot,
        uint256 recordedAt
    );
    event CheckpointAnchored(
        bytes32 indexed runId,
        uint32 checkpointIndex,
        bytes32 stateHash,
        uint256 anchoredAt
    );
    event CuratorUpdated(address indexed previousCurator, address indexed newCurator);
    event FeeCollectorUpdated(address indexed previousCollector, address indexed newCollector);
    event AnchorFeeSet(uint256 previousFeeWei, uint256 newFeeWei);
    event RunArchived(bytes32 indexed runId, address indexed archivedBy, uint256 atBlock);
    event TreasuryPull(address indexed to, uint256 amountWei, uint256 atBlock);

    // -------------------------------------------------------------------------
    // ERRORS
    // -------------------------------------------------------------------------

    error TP08_NotCurator();
    error TP08_ZeroAddress();
    error TP08_RunNotFound();
    error TP08_RunAlreadyArchived();
    error TP08_EpochIndexOutOfRange();
    error TP08_InvalidRunId();
    error TP08_InvalidConfigHash();
    error TP08_AnchorFeeRequired();
    error TP08_TransferFailed();
    error TP08_Reentrancy();
    error TP08_Paused();
    error TP08_EpochCountMismatch();
    error TP08_CheckpointIndexOutOfRange();
    error TP08_AlreadyInitialized();
    error TP08_InvalidFee();

    // -------------------------------------------------------------------------
    // CONSTANTS
    // -------------------------------------------------------------------------

    uint256 public constant TP08_VERSION = 8;
    uint256 public constant MAX_EPOCHS_PER_RUN = 50000;
    uint256 public constant MAX_CHECKPOINTS_PER_RUN = 2000;
    uint256 public constant LOSS_SCALE_FACTOR = 1e12;
    bytes32 public constant TP08_DOMAIN = keccak256("TensorProxima_08.Run.v8");

    // -------------------------------------------------------------------------
    // IMMUTABLES
    // -------------------------------------------------------------------------

    address public immutable curatorHub;
    address public immutable feeCollector;
    uint256 public immutable anchorFeeWei;

    // -------------------------------------------------------------------------
    // STATE
    // -------------------------------------------------------------------------

    struct TrainingRun {
        address submitter;
        uint16 epochCount;
        bytes32 configHash;
        uint256 registeredAt;
        bool archived;
        uint32 epochsRecorded;
        uint32 checkpointsAnchored;
    }

    mapping(bytes32 => TrainingRun) private _runs;
    mapping(bytes32 => mapping(uint32 => uint256)) private _epochLossScaled;
    mapping(bytes32 => mapping(uint32 => bytes32)) private _epochGradientRoot;
    mapping(bytes32 => mapping(uint32 => bytes32)) private _checkpointStateHash;
    address public curator;
    bool public paused;
    uint256 private _lock;
    bytes32[] private _runIdList;
    uint256 private _totalRuns;

