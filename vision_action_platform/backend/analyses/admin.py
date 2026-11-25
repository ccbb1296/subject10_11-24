from django.contrib import admin

from .models import AnalysisJob, PersonPrediction


class PersonPredictionInline(admin.TabularInline):
    """작업 상세 화면 안에서 사람별 결과를 같이 보기."""

    model = PersonPrediction
    extra = 0
    fields = ("track_id", "action_label", "confidence", "is_uncertain", "valid_frames", "window_count", "top3", "rule_note")


@admin.register(AnalysisJob)
class AnalysisJobAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "input_type", "status", "expected_action", "prediction_count", "created_at")
    list_filter = ("status", "input_type", "created_at")
    search_fields = ("user__username", "title", "memo", "celery_task_id")
    readonly_fields = ("created_at", "started_at", "completed_at", "celery_task_id", "processing_time_sec")
    inlines = [PersonPredictionInline]

    @admin.display(description="사람 수")
    def prediction_count(self, obj):
        return obj.predictions.count()


@admin.register(PersonPrediction)
class PersonPredictionAdmin(admin.ModelAdmin):
    list_display = ("id", "job", "track_id", "action_label", "confidence", "is_uncertain", "predicted_at")
    list_filter = ("action_label", "is_uncertain", "job__input_type")
    search_fields = ("job__user__username", "action_label")
