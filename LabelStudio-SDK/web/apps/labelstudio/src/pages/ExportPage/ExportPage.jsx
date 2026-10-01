import { useEffect, useRef, useState, useCallback } from "react";
import { useHistory } from "react-router";
import { Button, Badge } from "@humansignal/ui";
import {
  IconWarningCircleFilled,
  IconTerminal,
  IconCode,
  IconBook,
  IconExternal,
  IconCopyOutline,
} from "@humansignal/icons";
import { Form, Input } from "../../components/Form";
import { Modal } from "../../components/Modal/Modal";
import { Space } from "../../components/Space/Space";
import { useAPI } from "../../providers/ApiProvider";
import { useFixedLocation, useParams } from "../../providers/RoutesProvider";
import { cn } from "../../utils/bem";
import { isDefined, copyText } from "../../utils/helpers";
import "./ExportPage.prefix.css";

// Community Edition exports run synchronously in a single HTTP request.
// Large exports can exceed typical proxy timeouts, so we warn early and point at
// the snapshot export, which the server generates in the background.
const LARGE_EXPORT_TASK_THRESHOLD = 1000;
const EXPORT_TIMEOUT_DOCS_URL = "https://labelstud.io/guide/export.html#Export-timeout-in-Community-Edition";
const EXPORT_CONSOLE_DOCS_URL = "https://labelstud.io/guide/export.html#Export-using-console-command";
const EXPORT_SNAPSHOT_SDK_URL = "https://api.labelstud.io/api-reference/api-reference/projects/exports/create";
// Snapshot polling: every 5s, giving up after 2 hours so a snapshot that never
// leaves `in_progress` cannot poll forever. The export itself keeps running on
// the server either way.
const EXPORT_POLL_INTERVAL = 5000;
const EXPORT_POLL_ATTEMPTS = 1440;
const EXPORTS_LIST_LIMIT = 5;

const SNAPSHOT_STATUS_VARIANT = {
  completed: "positive",
  failed: "negative",
};
const SNAPSHOT_STATUS_LABEL = {
  created: "Queued",
  in_progress: "In progress",
  completed: "Ready",
  failed: "Failed",
};

// const formats = {
//   json: 'JSON',
//   csv: 'CSV',
// };

const downloadFile = (blob, filename) => {
  const link = document.createElement("a");

  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
};

const _wait = () => new Promise((resolve) => setTimeout(resolve, EXPORT_POLL_INTERVAL));

const isTimeoutLikeStatus = (status) => status === 408 || status === 502 || status === 504;

export const ExportPage = () => {
  const history = useHistory();
  const location = useFixedLocation();
  const pageParams = useParams();
  const api = useAPI();

  const [previousExports, setPreviousExports] = useState([]);
  const [downloading, setDownloading] = useState(false);
  const [downloadingMessage, setDownloadingMessage] = useState(false);
  const [availableFormats, setAvailableFormats] = useState([]);
  const [currentFormat, setCurrentFormat] = useState("JSON");
  const [projectTaskNumber, setProjectTaskNumber] = useState(null);
  const [exportIssue, setExportIssue] = useState(null);
  // Snapshot export. `snapshot` is non-null while one is being followed; the
  // work happens on the server, so neither it nor `creating` may block closing
  // the modal. `creating` only covers the create request, so the button stops
  // spinning as soon as the server has accepted the job.
  const [snapshot, setSnapshot] = useState(null);
  const [creating, setCreating] = useState(false);
  const [snapshotError, setSnapshotError] = useState(null);
  // Invalidates an in-flight poll loop when a new one starts.
  const pollToken = useRef(0);

  /** @type {import('react').RefObject<Form>} */
  const form = useRef();

  const proceedExport = async () => {
    setExportIssue(null);
    setDownloading(true);

    const messageTimer = window.setTimeout(() => {
      setDownloadingMessage(true);
    }, 1000);

    try {
      const params = form.current.assembleFormData({
        asJSON: true,
        full: true,
        booleansAsNumbers: true,
      });

      const response = await api.callApi("exportRaw", {
        params: {
          pk: pageParams.id,
          ...params,
        },
      });

      // The API proxy can return `null` for certain network errors; treat it as timeout-like
      // and show actionable guidance instead of a generic error.
      if (!response) {
        setExportIssue("timeout");
        return;
      }

      if (response.ok) {
        const blob = await response.blob();

        downloadFile(blob, response.headers.get("filename"));
        return;
      }

      if (isTimeoutLikeStatus(response.status)) {
        setExportIssue("timeout");
        return;
      }

      api.handleError(response);
    } finally {
      window.clearTimeout(messageTimer);
      setDownloading(false);
      setDownloadingMessage(false);
    }
  };

  const refreshExports = useCallback(async () => {
    const exports = await api.callApi("exportsList", {
      params: { pk: pageParams.id },
      errorFilter: () => true,
    });

    if (Array.isArray(exports)) setPreviousExports(exports);
  }, [api, pageParams.id]);

  // `exportType` is only passed for an export the user just asked for: the
  // server converts to the requested format on download, which is worth waiting
  // for there but not when someone clicks Download on an old snapshot.
  const downloadSnapshot = useCallback(
    async (record, exportType) => {
      const response = await api.callApi("exportDownloadRaw", {
        params: {
          pk: pageParams.id,
          exportPk: record.id,
          ...(exportType ? { exportType } : {}),
        },
      });

      if (!response) {
        setExportIssue("timeout");
        return;
      }
      if (!response.ok) {
        api.handleError(response);
        return;
      }

      downloadFile(await response.blob(), response.headers.get("filename") || `export-${record.id}.zip`);
    },
    [api, pageParams.id],
  );

  const trackSnapshot = useCallback(
    async (exportId) => {
      const token = ++pollToken.current;

      try {
        for (let attempt = 0; attempt < EXPORT_POLL_ATTEMPTS; attempt++) {
          await _wait();
          if (pollToken.current !== token) return;

          const record = await api.callApi("exportDetail", {
            params: { pk: pageParams.id, exportPk: exportId },
            errorFilter: () => true,
          });
          if (pollToken.current !== token) return;

          if (!record) {
            setSnapshotError("Lost track of this export. Reload the page to see its status.");
            return;
          }

          setSnapshot(record);

          if (record.status === "completed") {
            await downloadSnapshot(record, currentFormat);
            return;
          }
          if (record.status === "failed") {
            setSnapshotError(record.counters?.error || "The export failed on the server.");
            return;
          }
        }

        setSnapshotError("This export is taking longer than expected. It is still running on the server.");
      } finally {
        // Back to the normal view: the export is either downloaded, failed, or
        // still running server-side and listed under recent exports.
        if (pollToken.current === token) {
          setSnapshot(null);
          await refreshExports();
        }
      }
    },
    [api, pageParams.id, currentFormat, downloadSnapshot, refreshExports],
  );

  const proceedBackgroundExport = async () => {
    setExportIssue(null);
    setSnapshotError(null);
    setSnapshot(null);
    setCreating(true);

    let record;

    try {
      record = await api.callApi("createExport", {
        params: { pk: pageParams.id },
        // Ask the server to build it off the request thread. Without this the
        // POST would not return until the whole export was written, which is the
        // timeout this button exists to avoid.
        body: { background: true },
      });
    } finally {
      setCreating(false);
    }

    if (!record?.id) return;

    setSnapshot(record);
    // Followed in the background on purpose: the export runs on the server, so
    // there is no reason to keep this dialog busy (or open) until it finishes.
    trackSnapshot(record.id);
  };

  useEffect(() => {
    if (isDefined(pageParams.id)) {
      let cancelled = false;

      refreshExports();

      api
        .callApi("exportFormats", {
          params: {
            pk: pageParams.id,
          },
        })
        .then((formats) => {
          if (cancelled) return;
          setAvailableFormats(formats);
          setCurrentFormat(formats[0]?.name);
        });

      // Fetch project metadata to show a proactive warning for large exports.
      // This is best-effort and should not trigger global error UI if it fails.
      api
        .callApi("project", {
          params: { pk: pageParams.id },
          errorFilter: () => true,
        })
        .then((project) => {
          if (cancelled) return;
          setProjectTaskNumber(project?.task_number ?? null);
        });

      return () => {
        cancelled = true;
      };
    }
  }, [pageParams.id]);

  return (
    <Modal
      onHide={() => {
        const path = location.pathname.replace(ExportPage.path, "");
        const search = location.search;

        history.replace(`${path}${search !== "?" ? search : ""}`);
      }}
      title="Export data"
      style={{ width: 720 }}
      closeOnClickOutside={false}
      allowClose={!downloading}
      // footer="Read more about supported export formats in the Documentation."
      visible
    >
      <div className={cn("export-page").toClassName()}>
        <FormatInfo
          availableFormats={availableFormats}
          selected={currentFormat}
          onClick={(format) => setCurrentFormat(format.name)}
        />

        <ExportLargeProjectWarning taskCount={projectTaskNumber} />
        {exportIssue === "timeout" && <ExportTimeoutGuidance projectId={pageParams.id} exportType={currentFormat} />}

        {snapshot && (
          <div className={cn("export-page").elem("status-message").toClassName()}>
            Building the export on the server. You can close this window — it keeps running, and the finished file is
            downloaded for you.
          </div>
        )}
        {snapshotError && <div className={cn("export-page").elem("snapshot-error").toClassName()}>{snapshotError}</div>}

        {!snapshot && previousExports.length > 0 && (
          <SnapshotList exports={previousExports.slice(0, EXPORTS_LIST_LIMIT)} onDownload={downloadSnapshot} />
        )}

        <Form ref={form}>
          <Input type="hidden" name="exportType" value={currentFormat} />
        </Form>

        <div className={cn("export-page").elem("footer").toClassName()}>
          {downloadingMessage && (
            <div className={cn("export-page").elem("status-message").toClassName()}>
              Files are being prepared. It might take long time.
            </div>
          )}
          <Space style={{ width: "100%" }} spread>
            <div className={cn("export-page").elem("recent").toClassName()}>
              <a className="no-go" href={EXPORT_TIMEOUT_DOCS_URL} target="_blank" rel="noreferrer">
                Having a timeout or trouble exporting large projects?
              </a>
            </div>
            <div className={cn("export-page").elem("actions").toClassName()}>
              <Button
                look="outlined"
                variant="neutral"
                onClick={proceedBackgroundExport}
                waiting={creating}
                aria-label="Export in background"
              >
                Export in background
              </Button>
              <Button className="w-[135px]" onClick={proceedExport} waiting={downloading} aria-label="Export data">
                Export
              </Button>
            </div>
          </Space>
        </div>
      </div>
    </Modal>
  );
};

const FormatInfo = ({ availableFormats, selected, onClick }) => {
  return (
    <div className={cn("formats").toClassName()}>
      <div className={cn("formats").elem("info").toClassName()}>
        You can export dataset in one of the following formats:
      </div>
      <div className={cn("formats").elem("list").toClassName()}>
        {availableFormats.map((format) => (
          <div
            key={format.name}
            className={cn("formats")
              .elem("item")
              .mod({
                active: !format.disabled,
                selected: format.name === selected,
              })
              .toClassName()}
            onClick={!format.disabled ? () => onClick(format) : null}
          >
            <div className={cn("formats").elem("name").toClassName()}>
              {format.title}

              <Space size="small">
                {format.tags?.map?.((tag, index) => {
                  // Map tag text to badge variant
                  const tagLower = tag?.toLowerCase() || "";
                  let variant = "primary";
                  if (tagLower === "enterprise" || tagLower.includes("enterprise")) {
                    variant = "gradient";
                  } else if (tagLower === "beta") {
                    variant = "plum";
                  } else if (tagLower === "new" || tagLower.includes("new")) {
                    variant = "positive";
                  }

                  return (
                    <Badge key={index} variant={variant} size="small">
                      {tag}
                    </Badge>
                  );
                })}
              </Space>
            </div>

            {format.description && (
              <div className={cn("formats").elem("description").toClassName()}>{format.description}</div>
            )}
          </div>
        ))}
      </div>
      <div className={cn("formats").elem("feedback").toClassName()}>
        Can't find an export format?
        <br />
        Please let us know in{" "}
        <a className="no-go" href="https://slack.labelstud.io/?source=product-export" target="_blank" rel="noreferrer">
          Slack
        </a>{" "}
        or submit an issue to the{" "}
        <a
          className="no-go"
          href="https://github.com/HumanSignal/label-studio-converter/issues"
          target="_blank"
          rel="noreferrer"
        >
          Repository
        </a>
      </div>
    </div>
  );
};

ExportPage.path = "/export";
ExportPage.modal = true;

const SnapshotList = ({ exports, onDownload }) => {
  return (
    <div className={cn("export-page").elem("snapshots").toClassName()}>
      <div className={cn("export-page").elem("snapshots-title").toClassName()}>Recent exports</div>
      <ul className={cn("export-page").elem("snapshots-list").toClassName()}>
        {exports.map((item) => (
          <li key={item.id} className={cn("export-page").elem("snapshot").toClassName()}>
            <span className={cn("export-page").elem("snapshot-title").toClassName()} title={item.title}>
              {item.title}
            </span>
            <Badge size="small" variant={SNAPSHOT_STATUS_VARIANT[item.status] ?? "neutral"}>
              {SNAPSHOT_STATUS_LABEL[item.status] ?? item.status}
            </Badge>
            {item.status === "completed" && (
              <Button
                size="smaller"
                look="string"
                onClick={() => onDownload(item)}
                aria-label={`Download ${item.title}`}
              >
                Download
              </Button>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
};

const ExportLargeProjectWarning = ({ taskCount }) => {
  if (!Number.isFinite(taskCount) || taskCount < LARGE_EXPORT_TASK_THRESHOLD) return null;

  return (
    <div className={cn("export-page").elem("warning").toClassName()}>
      <div className={cn("export-page").elem("warning-title").toClassName()}>
        Large project detected ({taskCount.toLocaleString()} tasks)
      </div>
      <div className={cn("export-page").elem("warning-body").toClassName()}>
        Exporting this many tasks in one request can hit a proxy timeout. Use <strong>Export in background</strong>{" "}
        below: the server builds the file and you download it when it is ready. The{" "}
        <a className="no-go" href={EXPORT_TIMEOUT_DOCS_URL} target="_blank" rel="noreferrer">
          CLI/SDK export options
        </a>{" "}
        work too.
      </div>
    </div>
  );
};

const ExportTimeoutGuidance = ({ projectId, exportType }) => {
  const cliCommand = `label-studio export ${projectId} ${exportType} --export-path=<output-path>`;
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    copyText(cliCommand);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [cliCommand]);

  return (
    <div className={cn("export-page").elem("timeout").toClassName()}>
      <div className={cn("export-page").elem("timeout-header").toClassName()}>
        <IconWarningCircleFilled className={cn("export-page").elem("timeout-icon").toClassName()} />
        <div className={cn("export-page").elem("timeout-title").toClassName()}>Export timed out</div>
      </div>
      <div className={cn("export-page").elem("timeout-body").toClassName()}>
        This export was processed synchronously, in a single request, and can exceed typical reverse-proxy timeouts
        (often around 90 seconds) for large datasets. Use <strong>Export in background</strong> instead — the server
        builds the file and it is downloaded when ready.
      </div>

      <div className={cn("export-page").elem("timeout-actions").toClassName()}>
        <div className={cn("export-page").elem("timeout-actions-title").toClassName()}>Recommended options:</div>
        <ul className={cn("export-page").elem("timeout-actions-list").toClassName()}>
          <li>
            <div className={cn("export-page").elem("timeout-action-item").toClassName()}>
              <IconTerminal className={cn("export-page").elem("timeout-action-icon").toClassName()} />
              <div className={cn("export-page").elem("timeout-action-content").toClassName()}>
                <span>
                  Export using the{" "}
                  <a className="no-go" href={EXPORT_CONSOLE_DOCS_URL} target="_blank" rel="noreferrer">
                    console command
                    <IconExternal className={cn("export-page").elem("timeout-link-icon").toClassName()} />
                  </a>
                  :
                </span>
                <div className={cn("export-page").elem("timeout-code-wrapper").toClassName()}>
                  <pre className={cn("export-page").elem("timeout-code").toClassName()}>
                    <code>{cliCommand}</code>
                  </pre>
                  <button
                    type="button"
                    className={cn("export-page").elem("timeout-copy-button").toClassName()}
                    onClick={handleCopy}
                    aria-label="Copy command"
                    title={copied ? "Copied!" : "Copy command"}
                  >
                    <IconCopyOutline className={cn("export-page").elem("timeout-copy-icon").toClassName()} />
                    {copied && (
                      <span className={cn("export-page").elem("timeout-copy-text").toClassName()}>Copied</span>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </li>
          <li>
            <div className={cn("export-page").elem("timeout-action-item").toClassName()}>
              <IconCode className={cn("export-page").elem("timeout-action-icon").toClassName()} />
              <div className={cn("export-page").elem("timeout-action-content").toClassName()}>
                Use{" "}
                <a className="no-go" href={EXPORT_SNAPSHOT_SDK_URL} target="_blank" rel="noreferrer">
                  export snapshots via the SDK
                  <IconExternal className={cn("export-page").elem("timeout-link-icon").toClassName()} />
                </a>{" "}
                to create and download a snapshot without relying on a single UI request.
              </div>
            </div>
          </li>
          <li>
            <div className={cn("export-page").elem("timeout-action-item").toClassName()}>
              <IconWarningCircleFilled className={cn("export-page").elem("timeout-action-icon").toClassName()} />
              <div className={cn("export-page").elem("timeout-action-content").toClassName()}>
                Or press <strong>Export in background</strong> in this dialog: the snapshot is generated on the server
                and downloaded once it is ready, so no single request has to stay open.
              </div>
            </div>
          </li>
        </ul>
        <div className={cn("export-page").elem("timeout-footer").toClassName()}>
          <IconBook className={cn("export-page").elem("timeout-footer-icon").toClassName()} />
          <span>
            More details in the documentation:{" "}
            <a className="no-go" href={EXPORT_TIMEOUT_DOCS_URL} target="_blank" rel="noreferrer">
              Export timeout in Community Edition
              <IconExternal className={cn("export-page").elem("timeout-link-icon").toClassName()} />
            </a>
          </span>
        </div>
      </div>
    </div>
  );
};
