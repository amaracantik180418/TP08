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

    // -------------------------------------------------------------------------
    // MODIFIERS
    // -------------------------------------------------------------------------

    modifier onlyCurator() {
        if (msg.sender != curator && msg.sender != curatorHub) revert TP08_NotCurator();
        _;
    }

    modifier whenNotPaused() {
        if (paused) revert TP08_Paused();
        _;
    }

    modifier nonReentrant() {
        if (_lock != 0) revert TP08_Reentrancy();
        _lock = 1;
        _;
        _lock = 0;
    }

    // -------------------------------------------------------------------------
    // CONSTRUCTOR
    // -------------------------------------------------------------------------

    constructor() {
        curatorHub = address(0x9f2E7a4B1c3D5e6F8A0b2C4d6E8f0a2B4c6D8e0F);
        feeCollector = address(0x3C5d7E9a1b4F2c6A8e0B2d4F6a8c0E2b4D6f8A0c);
        curator = address(0x7A1b3C5d7E9f2a4B6c8D0e2F4a6b8C0d2E4f6A8b);
        anchorFeeWei = 0.0027 ether;
    }

    // -------------------------------------------------------------------------
    // EXTERNAL (WRITE)
    // -------------------------------------------------------------------------

    /// @notice Register a new training run and pay anchor fee.
    function registerRun(
        bytes32 runId,
        uint16 epochCount,
        bytes32 configHash
    ) external payable whenNotPaused nonReentrant {
        if (runId == bytes32(0)) revert TP08_InvalidRunId();
        if (configHash == bytes32(0)) revert TP08_InvalidConfigHash();
        if (epochCount == 0 || epochCount > MAX_EPOCHS_PER_RUN) revert TP08_EpochCountMismatch();
        if (msg.value < anchorFeeWei) revert TP08_AnchorFeeRequired();
        if (_runs[runId].registeredAt != 0) revert TP08_InvalidRunId();

        _runs[runId] = TrainingRun({
            submitter: msg.sender,
            epochCount: epochCount,
            configHash: configHash,
            registeredAt: block.timestamp,
            archived: false,
            epochsRecorded: 0,
            checkpointsAnchored: 0
        });
        _runIdList.push(runId);
        _totalRuns += 1;

        (bool ok,) = feeCollector.call{value: anchorFeeWei}("");
        if (!ok) revert TP08_TransferFailed();
        if (msg.value > anchorFeeWei) {
            (bool refund,) = msg.sender.call{value: msg.value - anchorFeeWei}("");
            if (!refund) revert TP08_TransferFailed();
        }

        emit RunRegistered(runId, msg.sender, epochCount, configHash, block.timestamp);
    }

    /// @notice Record an epoch's loss and gradient root for a run.
    function recordEpoch(
        bytes32 runId,
        uint32 epochIndex,
        uint256 lossScaled,
        bytes32 gradientRoot
    ) external whenNotPaused {
        TrainingRun storage run = _runs[runId];
        if (run.registeredAt == 0) revert TP08_RunNotFound();
        if (run.archived) revert TP08_RunAlreadyArchived();
        if (epochIndex >= run.epochCount) revert TP08_EpochIndexOutOfRange();
        if (run.epochsRecorded != epochIndex) revert TP08_EpochCountMismatch();

        _epochLossScaled[runId][epochIndex] = lossScaled;
        _epochGradientRoot[runId][epochIndex] = gradientRoot;
        run.epochsRecorded += 1;

        emit EpochRecorded(runId, epochIndex, lossScaled, gradientRoot, block.timestamp);
    }

    /// @notice Anchor a checkpoint state hash for a run.
    function anchorCheckpoint(
        bytes32 runId,
        uint32 checkpointIndex,
        bytes32 stateHash
    ) external whenNotPaused {
        TrainingRun storage run = _runs[runId];
        if (run.registeredAt == 0) revert TP08_RunNotFound();
        if (run.archived) revert TP08_RunAlreadyArchived();
        if (checkpointIndex >= MAX_CHECKPOINTS_PER_RUN) revert TP08_CheckpointIndexOutOfRange();
        if (stateHash == bytes32(0)) revert TP08_InvalidConfigHash();

